clear;
resultfile_list = {
'Y:\TFT_based_PFU\TFT_PFU_Network_Code_Yuzhu\Github\Example_Data\Output\log_20251121_8diffchannel_timeframe8_32_24_b3363_dr0.5_center50'
};

for kkkk = 1:length(resultfile_list)
    resultfile = resultfile_list{kkkk};
    alltimelist = dir(fullfile(resultfile,'result_time*_epoch277.txt'));
    x1 = 96;
    x2 = 96;
    circthreshold = 0.7; 
    decision_threshold = 0.5; 
    start_time = 1;

    %
    [xx, yy] = meshgrid(-x1/2+1:x1/2,-x2/2+1:x2/2);
    x_shift = 0; % can be tuned to center the FOVs
    y_shift = 0;
    r = 40; % unit:mm previous: 40
    effect_area = ((xx-x_shift).^2+(yy-y_shift).^2) < r^2;

    % initialization of count setting
    L_prev = zeros(x1,x2);
    persistent_centers_list = [];
    PFU_count= zeros(1,length(alltimelist)+14);
    PFU_count_diff = zeros(1,length(alltimelist)+14);
    PFU_Area_per_PFU = zeros(1,length(alltimelist)+14);
    Total_PFU_Area = zeros(1,length(alltimelist)+14); 
    PFU_Growth_Matrix = nan(0, 0); 
    PFU_Birth_Times = []; 
    
    for kkk = start_time:length(alltimelist)
        kkk
        areathreshod = 5; 
 
        curfile = fullfile(resultfile,alltimelist(kkk).name);
        fid = fopen(curfile, "r");
    
        while feof(fid) ~= 1
              resultTXT = fgetl(fid);
              a = sscanf(resultTXT,'%f %f %f');
              detectionNNscore(a(1),:) = a(2:3); % 1:negative 2:positive
        end
        fclose(fid);
        
        clear problocal
        kk = 1;
        for ii = 1:x1
            for jj = 1:x2
                problocal(ii,jj) = detectionNNscore(kk,2);
                kk = kk+1;
            end
        end
    
        problocal_raw(:,:,kkk) = problocal.*effect_area;
        problocal_abovethr(:,:,kkk) = (problocal_raw(:,:,kkk)>decision_threshold);
    
         % delete very small isolated regions
        CC = bwconncomp(problocal_abovethr(:,:,kkk),4);
        stats = regionprops(CC, 'Area', 'BoundingBox', 'Perimeter');

        Arealist = cat(1, stats(:).Area);
        Premiterlist = cat(1, stats(:).Perimeter);
        Circularitylist = (4 * pi .* Arealist ./ (Premiterlist).^2)...
            .*((1-0.5./(Premiterlist/2/pi+0.5)).^2);
        idx_delete = (Arealist<areathreshod) | (Circularitylist<circthreshold);
        stats_delete = stats(idx_delete);
        Boundingboxlist = cat(1, stats_delete(:).BoundingBox);
        for del = 1:length(stats_delete)
            current_box = Boundingboxlist(del,:);
            if floor(current_box(2)) >= 1 & floor(current_box(1)) >= 1
                problocal_abovethr(floor(current_box(2)):floor(current_box(2)+current_box(4)),floor(current_box(1)):floor(current_box(1)+current_box(3)),kkk) = 0;
            end
        end

        problocal_max(:,:,kkk) = prctile(problocal_raw(:,:,start_time:kkk).*problocal_abovethr(:,:,start_time:kkk),100,3);
        
        % correct the distortion (updated on 2025/11/11)
        img_toshow = correctPincushion(problocal_raw(:,:,kkk),0.5);
        mask_toshow = correctPincushion(problocal_abovethr(:,:,kkk),0.5);
        max_toshow = correctPincushion(problocal_max(:,:,kkk),0.5);

%         img_toshow = problocal_raw(:,:,kkk);
%         mask_toshow = problocal_abovethr(:,:,kkk);
%         max_toshow = problocal_max(:,:,kkk);

        % statistics
        % Metric 1: PFU count
        final_detection(:,:,kkk) =  max_toshow >= decision_threshold;
        final_CC = bwconncomp(final_detection(:,:,kkk),4); % using cleaned binary map
        
        final_Label = labelmatrix(final_CC);
        final_stats = regionprops(final_CC, 'Centroid');
        num_temp_objects = final_CC.NumObjects;

        num_existing_centers = size(persistent_centers_list, 1);

        for i = 1:num_temp_objects
            current_pixel_list = final_CC.PixelIdxList{i};
            overlapping_labels_vector = L_prev(current_pixel_list);
            unique_prev_labels = unique(overlapping_labels_vector(overlapping_labels_vector > 0));
            
            centers_to_add = []; 
            if isempty(unique_prev_labels)
                % case 1: new PFU
                centers_to_add = [centers_to_add; final_stats(i).Centroid];         
            else
                % case 2: old pfu growth or new PFU very close to old pfu
                current_mask_2D = false(size(L_prev));
                current_mask_2D(current_pixel_list) = true;
                
                new_growth_mask = (current_mask_2D == 1) & (L_prev == 0);

                if ~any(new_growth_mask(:))
                    continue; 
                end
                
                CC_new_growth = bwconncomp(new_growth_mask);
                
                if CC_new_growth.NumObjects == 0
                    continue;
                end
                
                stats_new_growth = regionprops(CC_new_growth, 'Centroid', 'Area');
                
                for k = 1:CC_new_growth.NumObjects
                    if stats_new_growth(k).Area >= 5
                        centers_to_add = [centers_to_add; stats_new_growth(k).Centroid];
                    end
                end
            end
            
            if ~isempty(centers_to_add)
                num_new = size(centers_to_add, 1);
                persistent_centers_list = [persistent_centers_list; centers_to_add];
                
                
                PFU_Birth_Times = [PFU_Birth_Times; repmat(kkk, num_new, 1)];
                
               
                current_cols = size(PFU_Growth_Matrix, 2);
                PFU_Growth_Matrix = [PFU_Growth_Matrix; nan(num_new, current_cols)];
            end
        end
        
        num_total_pfus = size(persistent_centers_list, 1);
        PFU_count(kkk+14) = num_total_pfus;

        if num_total_pfus > 0
        
            [rows, cols] = find(final_detection(:,:,kkk));
            all_pixel_coords = [cols, rows]; 
            
            if ~isempty(all_pixel_coords)
                
                nearest_idx = dsearchn(persistent_centers_list, all_pixel_coords);
               
                current_areas = accumarray(nearest_idx, 1, [num_total_pfus, 1]);
            else
                current_areas = zeros(num_total_pfus, 1);
            end
            
            for p_idx = 1:num_total_pfus
                birth_time = PFU_Birth_Times(p_idx);
                
                relative_time_idx = kkk - birth_time + 1;
                
                if relative_time_idx > 0
                    if relative_time_idx > size(PFU_Growth_Matrix, 2)
                        cols_to_add = relative_time_idx - size(PFU_Growth_Matrix, 2);
                        PFU_Growth_Matrix = [PFU_Growth_Matrix, nan(size(PFU_Growth_Matrix, 1), cols_to_add)];
                    end
                    
                    PFU_Growth_Matrix(p_idx, relative_time_idx) = current_areas(p_idx);
                end
            end
        end
        average_area_per_delta_time = mean(PFU_Growth_Matrix, 1, 'omitnan');

        % Metric 1-2: newly grown pfu count per hour
        PFU_count_diff(kkk+14) = PFU_count(kkk+14) - PFU_count(kkk+13);
        
        L_prev = final_Label;
        
        % Metric 2: Total Area
        Total_PFU_Area(kkk+14) = sum(final_detection(:,:,kkk),'all');

        % Metric 3: Area per PFU
        if PFU_count(kkk+14) ~= 0
            PFU_Area_per_PFU(kkk+14) = Total_PFU_Area(kkk+14) / PFU_count(kkk+14);
        else
            PFU_Area_per_PFU(kkk+14) = 0;
        end

        % save final results
        h1 = figure;imagesc(imresize(img_toshow,1),[0,1]);axis off;axis square;
        saveas(h1,fullfile(resultfile,replace(alltimelist(kkk).name,'.txt','_prob_corrected_test.tiff')));
        close(h1);
    
%         h2 = figure;imagesc(imresize(mask_toshow,1),[0,1]);axis off;axis square;
%         saveas(h2,fullfile(resultfile,replace(alltimelist(kkk).name,'.txt','_mask_corrected.tiff')));
%         close(h2);

        h3 = figure;imagesc(imresize(max_toshow,1),[0,1]);axis off;axis square;
        saveas(h3,fullfile(resultfile,replace(alltimelist(kkk).name,'.txt','_max_corrected_test.tiff')));
        close(h3);

        mask = single(final_detection(:,:,kkk));
        mask(mask==0) = 0.5;
        h4 = figure;imshow(imresize(mask,5),[0,1]);axis off;axis square;
        hold on;
        if ~isempty(persistent_centers_list)
            scatter(persistent_centers_list(:,1)*5,persistent_centers_list(:,2)*5,'r+','LineWidth',1.5);
        end
        saveas(h4,fullfile(resultfile,replace(alltimelist(kkk).name,'.txt','_final_detection_binary_test.tiff')));
        close(h4);
    end
    save(fullfile(resultfile,'final_statics_test.mat'), 'PFU_count', 'PFU_Area_per_PFU', 'Total_PFU_Area','PFU_count_diff','PFU_Growth_Matrix','average_area_per_delta_time');
end