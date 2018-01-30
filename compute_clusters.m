clear all

%% paths
set_proposal_flow_path;
%input folder which is the output of the first part of the method (mask creation)
in_base_folder = '/media/vips/data/mgodi/input_classes_dense/';
out_base_folder = '/media/vips/data/mgodi/output_classes/';


class_folders = dir([in_base_folder '*']);
class_folders = class_folders(3:end);
mat_files = dir([out_base_folder '*.mat']); %to precompute results; to disable saving mat files, comment last save line at the end of the code

for class_ind=1:numel(class_folders)
    '***********************************************************************************************'
    class_folders = dir([in_base_folder '*']);
    class_folders = class_folders(3:end);
    class_folders(class_ind).name
    tstart=tic;
    base_path = fullfile(class_folders(class_ind).folder, class_folders(class_ind).name);
    out_folder = fullfile(out_base_folder, class_folders(class_ind).name);
    if exist(out_folder)
        % delete already existing folder to recompute output
        continue
    end
    mkdir(out_folder);
    if exist([fullfile(out_base_folder, class_folders(class_ind).name) '.mat'])
        %load existing mat file
        load([fullfile(out_base_folder, class_folders(class_ind).name) '.mat']);
        [class_folders(class_ind).name '.mat']
        tstart=tic;
    else
        
        base_path = fullfile(class_folders(class_ind).folder, class_folders(class_ind).name);

        %% config to set
        num_op = 500; %number of region proposals (rps) per image
        img_numb = 100; %number of images to process (suggested number=100) (NOTE: time increases quadratically with the number of images)
        valid_rp_thresh = 0.75; %threshold for the mask overlap, to consider a region proposal as valid
        same_region_thresh = 0.5; %threhsold of overlap to consider two rps redundant
        scale_factor_thresh = 2.0; %threhsold of relative scale difference to consider 2 rps of the same scale

        doWarp = false;


        img_paths = dir(fullfile(base_path, '*/smooth/original.png'));
        sharp_mask_paths = dir(fullfile(base_path, '*/sharp/mask.png'));

        img_numb = min(img_numb, numel(img_paths));
        img_paths = img_paths(1:img_numb);
        sharp_mask_paths = sharp_mask_paths(1:img_numb);
        


        %% create proposals
        proposals = cell(numel(img_paths), 1);

        good_inds = false(numel(img_paths), 1);

        count = 0;
        'creating proposals'
        parfor img_ind=1:numel(img_paths)
            tmp_pr = SS(imread(fullfile(img_paths(img_ind).folder, img_paths(img_ind).name)), num_op); %selective search
            if size(tmp_pr, 1) < num_op
                continue
            end

            proposals{img_ind} = tmp_pr;
            good_inds(img_ind) = true;
        end
        'ending proposal creations'
        good_inds = find(good_inds);
        img_paths = img_paths(good_inds);
        sharp_mask_paths = sharp_mask_paths(good_inds);
        proposals = proposals(good_inds);

        %% compute flow

        registrations = cell(numel(img_paths));
        'start registrations'
        parfor img_ind1=1:numel(img_paths)
            regrow = registrations(img_ind1, :);
            for img_ind2=1:numel(img_paths)
                if img_ind2 < img_ind1
                    continue
                end
                %compute proposal flow correspondence matrix between each
                %pair of images
                tmp_proposals = [proposals{img_ind1} ; proposals{img_ind2}];
                [match, weight, confidence, warp] = compute_flow_fn(fullfile(img_paths(img_ind1).folder, img_paths(img_ind1).name), fullfile(img_paths(img_ind2).folder, img_paths(img_ind2).name), num_op, tmp_proposals, doWarp);
                regrow{img_ind2} = confidence;
            end
            registrations(img_ind1, :) = regrow;
            img_ind1
        end
        
        for img_ind1=1:numel(img_paths)
            for img_ind2=1:numel(img_paths)
                if img_ind2 < img_ind1
                    registrations{img_ind1, img_ind2} = registrations{img_ind2, img_ind1}';
                end
            end
        end
        'end registrations'

        %% filter regions on mask better
        good_rps = cell(numel(img_paths), 1);
        total_rps = 0;
        for img_ind=1:numel(img_paths)
            
            % extract slice of similarity matrix for heuristics
            similarity_mat_part = zeros(num_op, 0);
            columns_written = 0;
            for img_ind2=1:numel(img_paths)
                if img_ind == img_ind2
                    continue
                end
                tmp_reg = registrations{img_ind, img_ind2};
                similarity_mat_part = [similarity_mat_part tmp_reg];
                columns_written = columns_written + num_op;
            end

            % read mask
            mask = imread(fullfile(sharp_mask_paths(img_ind).folder, sharp_mask_paths(img_ind).name));
            mask = mask > 128;
            tmp_props = proposals{img_ind};
            tmp_good_inds = [];
            for prop_ind=1:num_op
                % check how much each proposal overlaps with mask
                tmp_prop = tmp_props(prop_ind, :);
                tmp_maskpart = mask(tmp_prop(2):tmp_prop(4), tmp_prop(1):tmp_prop(3));
                area = numel(tmp_maskpart);
                if sum(tmp_maskpart(:)) > (area * valid_rp_thresh)
                    tmp_good_inds = [tmp_good_inds prop_ind];
                end
                %and remove if doesn't overlap enought
            end

            % remove redundant proposals (redundant if overlaps with another
            % region proposal)
            % keep the one with the best average score in sim matrix
            count = 0;
            while count < numel(tmp_good_inds)
                count = count + 1;
                good_ind1 = tmp_good_inds(count);
                tmp_bbox1 = [tmp_props(good_ind1, 1) tmp_props(good_ind1, 2) tmp_props(good_ind1, 3)-tmp_props(good_ind1, 1) tmp_props(good_ind1, 4)-tmp_props(good_ind1, 2)];
                to_keep = [];
                redundant_inds = [good_ind1];

                [tmp_sval, ~] = sort(similarity_mat_part(good_ind1, :),'descend');
                best_scores = [mean(tmp_sval(1:100))];
                for good_ind2=tmp_good_inds
                    if good_ind1 == good_ind2
                        continue
                    end
                    tmp_bbox2 = [tmp_props(good_ind2, 1) tmp_props(good_ind2, 2) tmp_props(good_ind2, 3)-tmp_props(good_ind2, 1) tmp_props(good_ind2, 4)-tmp_props(good_ind2, 2)];
                    oratio = bboxOverlapRatio(tmp_bbox1, tmp_bbox2, 'Min');
                    if oratio < same_region_thresh
                        to_keep = [to_keep good_ind2];
                    else
                        %redundant if they are of roughly the same size on both
                        %dimensions
                        sc1 = prod(tmp_bbox1(3:4));
                        sc2 = prod(tmp_bbox2(3:4));
                        sdist = max([sc1 sc2]) / min([sc1 sc2]);

        %                 %if one is a lot bigger keep them both
                        if sdist < scale_factor_thresh
                            redundant_inds = [redundant_inds good_ind2];

                            [tmp_sval, ~] = sort(similarity_mat_part(good_ind2, :),'descend');
                            best_scores = [best_scores mean(tmp_sval(1:100))];
                        else
                            to_keep = [to_keep good_ind2];
                        end
                    end
                end
                [~, best_ind] = max(best_scores);
                tmp_good_inds = [redundant_inds(best_ind) to_keep];
            end

            good_rps{img_ind} = tmp_good_inds;
            total_rps = total_rps + numel(tmp_good_inds);
        end
    end
    %% build similarity matrix

    similarity_mat = zeros(total_rps, total_rps);
    reg_img_inds = [];
    reg_rp_inds = [];
    rows_written = 0;
    for img_ind1=1:numel(img_paths)
        tmp_props = proposals{img_ind1};
        tmp_good_inds = good_rps{img_ind1};
        columns_written = 0;
        for img_ind2=1:numel(img_paths)
            % for each other image find registration scores for valid rps only
            tmp_reg = registrations{img_ind1, img_ind2};
            tmp_good_inds2 = good_rps{img_ind2};

            tmp_reg = tmp_reg(tmp_good_inds, tmp_good_inds2);
            similarity_mat(rows_written + 1:rows_written + numel(tmp_good_inds), columns_written + 1:columns_written + numel(tmp_good_inds2)) = tmp_reg;
            
            columns_written = columns_written + numel(tmp_good_inds2);
        end
       reg_img_inds = [reg_img_inds ones(1, numel(tmp_good_inds)) .* img_ind1];
       reg_rp_inds = [reg_rp_inds tmp_good_inds];
       rows_written = rows_written + numel(tmp_good_inds);
    end
    %% clustering
    similarity_mat_orig = similarity_mat;
    similarity_mat = similarity_mat - max(similarity_mat(:)); % 
    p = 1.3;
    [cls, ~, ~, ~] = apcluster(similarity_mat, median(similarity_mat(:)) * p); %-600); %affinity propagation
    fprintf('Numero cluster: %d \n', numel(unique(cls)));

    %% visualize cluster
    count_cl = 0;
    for cl=unique(cls)'
        cl_inds = find(cls == cl);
        cl_img_ind = reg_img_inds(cl);
        cl_rp_ind = reg_rp_inds(cl);
        
        %remove DIFFERENT region proposals from the same image as the representative 
        cl_inds = cl_inds(reg_img_inds(cl_inds) ~= cl_img_ind | (reg_img_inds(cl_inds) == cl_img_ind & reg_rp_inds(cl_inds) == cl_rp_ind));
        
        f = figure('visible','off');
        f.PaperUnits = 'inches';
        f.PaperPosition = [0 0 10 10];
        subplot(5,5,1)
        
        tmp_img = imread(fullfile(img_paths(cl_img_ind).folder, img_paths(cl_img_ind).name));
        tmp_props = proposals{cl_img_ind};
        
        imshow(tmp_img(tmp_props(cl_rp_ind, 2):tmp_props(cl_rp_ind, 4), tmp_props(cl_rp_ind, 1):tmp_props(cl_rp_ind, 3), :));
        
        %sort by similarity value
        sim_row = similarity_mat(cl, cl_inds');
        [sorted_sims, sorted_sim_inds] = sort(sim_row, 'descend');
        sorted_cl_inds = cl_inds(sorted_sim_inds);
        %have only one rp per image (no rps from same image, only the best one)
        [~, unique_sorted_sim_inds, ~] = unique(reg_img_inds(sorted_cl_inds), 'stable'); %OCCURENCE = first by default
        sorted_cl_inds = sorted_cl_inds(unique_sorted_sim_inds);
        
        for sp_ind=1:min(20, numel(sorted_cl_inds))
            cl_ind = sorted_cl_inds(sp_ind);
            subplot(5,5,sp_ind+5)
            cl_img_ind2 = reg_img_inds(cl_ind);
            tmp_img2 = imread(fullfile(img_paths(cl_img_ind2).folder, img_paths(cl_img_ind2).name));
            tmp_props2 = proposals{cl_img_ind2};
            cl_rp_ind2 = reg_rp_inds(cl_ind);
            imshow(tmp_img2(tmp_props2(cl_rp_ind2, 2):tmp_props2(cl_rp_ind2, 4), tmp_props2(cl_rp_ind2, 1):tmp_props2(cl_rp_ind2, 3), :));
            title(similarity_mat_orig(cl, cl_ind));
        end
        count_cl = count_cl + 1;
        print([out_folder '/cluster_' num2str(count_cl)],'-dpng', '-r0');
    end

    %% compute visual summaries
    close all
    count_cl = 0;
    topk = 10;
    blendrgbwarps = cell(numel(unique(cls)), 1);
    tot_cl = numel(unique(cls));
    cluster_exemplars = unique(cls);
    all_imgs_rgb_by_cl = cell(tot_cl, 1);
    contexts_and_convexhulls_by_cl = cell(tot_cl, 1);

    for cl_cursor=1:tot_cl
        cl = cluster_exemplars(cl_cursor);
        count_cl = find(cluster_exemplars == cl);
        count_cl = cl_cursor;
        
        cl_inds = find(cls == cl);
        cl_img_ind = reg_img_inds(cl);
        cl_rp_ind = reg_rp_inds(cl);

        %remove DIFFERENT region proposals from the same image as the representative 
        cl_inds = cl_inds(reg_img_inds(cl_inds) ~= cl_img_ind | (reg_img_inds(cl_inds) == cl_img_ind & reg_rp_inds(cl_inds) == cl_rp_ind));

        f = figure(cl_cursor);
        f.Visible = 'off';
        f.PaperUnits = 'inches';
        f.PaperPosition = [0 0 10 10];

        %read exemplar of current cluster
        tmp_img = imread(fullfile(img_paths(cl_img_ind).folder, img_paths(cl_img_ind).name));
        tmp_props = proposals{cl_img_ind};

        %cut on region proposal
        tmp_img = tmp_img(tmp_props(cl_rp_ind, 2):tmp_props(cl_rp_ind, 4), tmp_props(cl_rp_ind, 1):tmp_props(cl_rp_ind, 3), :);
        tmp_img_gray = rgb2gray(tmp_img);
        
        %sort by similarity values
        sim_row = similarity_mat(cl, cl_inds');
        [sorted_sims, sorted_sim_inds] = sort(sim_row, 'descend');
        sorted_cl_inds = cl_inds(sorted_sim_inds);
        %have only one rp per image (no rps from same image, only the best one)
        [~, unique_sorted_sim_inds, ~] = unique(reg_img_inds(sorted_cl_inds), 'stable'); %OCCURENCE = first by default
        sorted_cl_inds = sorted_cl_inds(unique_sorted_sim_inds);

        all_imgs_rgb = zeros(min(topk, numel(sorted_cl_inds)), size(tmp_img, 1), size(tmp_img, 2), 3);
        all_imgs_gray = zeros(min(topk, numel(sorted_cl_inds)), size(tmp_img, 1), size(tmp_img, 2), 1);
        all_imgs_mask = zeros(min(topk, numel(sorted_cl_inds)), size(tmp_img, 1), size(tmp_img, 2), 1);
        scores = zeros(min(topk, numel(sorted_cl_inds)), 1);

        count_sp_ind = 0;
        for sp_ind=1:min(topk, numel(sorted_cl_inds))
            cl_ind = sorted_cl_inds(sp_ind);
            cl_img_ind2 = reg_img_inds(cl_ind);
            tmp_img2 = imread(fullfile(img_paths(cl_img_ind2).folder, img_paths(cl_img_ind2).name));
            tmp_mask2 = imread(fullfile(sharp_mask_paths(cl_img_ind2).folder, sharp_mask_paths(cl_img_ind2).name));
            tmp_props2 = proposals{cl_img_ind2};
            cl_rp_ind2 = reg_rp_inds(cl_ind);

            %warp second image to exemplar (this is used to evaluate later
            %how good is a cluster)
            tmp_proposals = [tmp_props ; tmp_props2];
            [~, ~, ~, warp] = compute_flow_fn(fullfile(img_paths(cl_img_ind).folder, img_paths(cl_img_ind).name), fullfile(img_paths(cl_img_ind2).folder, img_paths(cl_img_ind2).name), num_op, tmp_proposals, true);
            tmp_img2_warped = warpImage(im2double(tmp_img2),warp.vx,warp.vy);
            tmp_img2_warped = tmp_img2_warped .* 255.0;
            tmp_mask2_warped = warpImage(im2double(tmp_mask2),warp.vx,warp.vy);

            tmp_img2 = tmp_img2_warped(tmp_props(cl_rp_ind, 2):tmp_props(cl_rp_ind, 4), tmp_props(cl_rp_ind, 1):tmp_props(cl_rp_ind, 3), :);
            tmp_mask2 = tmp_mask2_warped(tmp_props(cl_rp_ind, 2):tmp_props(cl_rp_ind, 4), tmp_props(cl_rp_ind, 1):tmp_props(cl_rp_ind, 3), :);
            
            %save warped images
            count_sp_ind = count_sp_ind + 1;
            all_imgs_rgb(count_sp_ind, :, :, :) = imresize(tmp_img2, [size(tmp_img, 1), size(tmp_img, 2)]);
            all_imgs_gray(count_sp_ind, :, :, :) = imresize(rgb2gray(tmp_img2), [size(tmp_img, 1), size(tmp_img, 2)]);
            all_imgs_mask(count_sp_ind, :, :, :) = imresize(tmp_mask2, [size(tmp_img, 1), size(tmp_img, 2)]);
            scores(count_sp_ind) = similarity_mat_orig(cl, cl_ind);
        end
        
        all_imgs_rgb_by_cl{count_cl} = all_imgs_rgb;

        %uniform scoring (unused)
        scores = ones(numel(scores), 1) ./ numel(scores);
    
        %visualize warped clusters
        n_subplots = ceil(sqrt(min(topk, numel(sorted_cl_inds))));

        all_imgs_rgb_orig = all_imgs_rgb;
        all_imgs_mask_orig = all_imgs_mask;
        
        all_imgs_rgb = all_imgs_rgb .* scores; %unused
        all_imgs_gray = all_imgs_gray .* scores;

        all_imgs_mask = squeeze(mean(all_imgs_mask, 1)) > 0.75;
        all_imgs_mask_ch = bwconvhull(all_imgs_mask); %convexhull image
        all_imgs_mask_properties = regionprops(all_imgs_mask_ch, 'ConvexHull');
        if size(all_imgs_mask_properties, 1) == 0
            continue
        end
        conv_coords = all_imgs_mask_properties(1).ConvexHull; %convexhull coords
        all_imgs_rgb_blend = uint8(squeeze(sum(all_imgs_rgb, 1)));
        
        %******************************
        %save info about summaries (image, convexhull of rp, etc)
        n_subplots = ceil(sqrt(size(all_imgs_rgb, 1)));
        
        contexts_and_convexhulls_by_cl{count_cl}.contexts = cell(size(all_imgs_rgb, 1), 1);
        contexts_and_convexhulls_by_cl{count_cl}.chrs = cell(size(all_imgs_rgb, 1), 1);
        contexts_and_convexhulls_by_cl{count_cl}.chcs = cell(size(all_imgs_rgb, 1), 1);
        contexts_and_convexhulls_by_cl{count_cl}.conveximgs = cell(size(all_imgs_rgb, 1), 1);
        contexts_and_convexhulls_by_cl{count_cl}.bboxes = cell(size(all_imgs_rgb, 1), 1);
        
        for ii=1:size(all_imgs_rgb, 1)
            subplot(n_subplots, n_subplots, ii);

            cl_ind = sorted_cl_inds(ii);
            cl_img_ind2 = reg_img_inds(cl_ind);
            tmp_img2 = imread(fullfile(img_paths(cl_img_ind2).folder, img_paths(cl_img_ind2).name));
            tmp_mask2 = imread(fullfile(sharp_mask_paths(cl_img_ind2).folder, sharp_mask_paths(cl_img_ind2).name)) > 1;
            tmp_props2 = proposals{cl_img_ind2};
            cl_rp_ind2 = reg_rp_inds(cl_ind);
            tmp_mask2_rp = tmp_mask2(tmp_props2(cl_rp_ind2, 2):tmp_props2(cl_rp_ind2, 4), tmp_props2(cl_rp_ind2, 1):tmp_props2(cl_rp_ind2, 3));
            %compute convex hull of rp on mask
            tmp_mask2_ch = bwconvhull(tmp_mask2_rp);
            tmp_mask2_properties = regionprops(tmp_mask2_ch, 'ConvexHull');
            conv_coords2 = tmp_mask2_properties(1).ConvexHull;
            
            contexts_and_convexhulls_by_cl{count_cl}.contexts{ii} = tmp_img2;
            contexts_and_convexhulls_by_cl{count_cl}.chrs{ii} = conv_coords2(:, 1) - 1 + tmp_props2(cl_rp_ind2, 1);
            contexts_and_convexhulls_by_cl{count_cl}.chcs{ii} = conv_coords2(:, 2) - 1 + tmp_props2(cl_rp_ind2, 2);
            tmp_chimg = zeros(size(tmp_mask2, 1), size(tmp_mask2, 2));
            tmp_chimg(tmp_props2(cl_rp_ind2, 2):tmp_props2(cl_rp_ind2, 4), tmp_props2(cl_rp_ind2, 1):tmp_props2(cl_rp_ind2, 3)) = tmp_mask2_ch;
            contexts_and_convexhulls_by_cl{count_cl}.conveximgs{ii} = tmp_chimg;
            contexts_and_convexhulls_by_cl{count_cl}.bboxes{ii} = tmp_props2(cl_rp_ind2, :);
        end
        
        %******************************
        %additional info being computed
        n_subplots = ceil(sqrt(size(all_imgs_rgb, 1)));
        
        
        contexts_and_convexhulls_by_cl{count_cl}.maskprops = cell(size(all_imgs_rgb, 1), 1);
        
        for ii=1:size(all_imgs_rgb, 1)

            cl_ind = sorted_cl_inds(ii);
            cl_img_ind2 = reg_img_inds(cl_ind);
            tmp_img2 = imread(fullfile(img_paths(cl_img_ind2).folder, img_paths(cl_img_ind2).name));
            tmp_mask2 = imread(fullfile(sharp_mask_paths(cl_img_ind2).folder, sharp_mask_paths(cl_img_ind2).name)) > 1;
            tmp_props2 = proposals{cl_img_ind2};
            cl_rp_ind2 = reg_rp_inds(cl_ind);
            tmp_mask2_rp = tmp_mask2(tmp_props2(cl_rp_ind2, 2):tmp_props2(cl_rp_ind2, 4), tmp_props2(cl_rp_ind2, 1):tmp_props2(cl_rp_ind2, 3));
            tmp_mask2_ch = bwconvhull(tmp_mask2_rp);
            tmp_mask2_properties = regionprops(tmp_mask2_ch, 'ConvexHull');
            conv_coords2 = tmp_mask2_properties(1).ConvexHull;

            tmp_mask2_properties = regionprops(tmp_mask2, 'Image', 'BoundingBox');
            
            contexts_and_convexhulls_by_cl{count_cl}.maskprops{ii} = tmp_mask2_properties;
        end

        blendrgbwarps{count_cl} = uint8(squeeze(sum(all_imgs_rgb, 1)));
    end
    close all
    
    
    
%% compute global measures of quality for this class
    mediane = [];
    medie = [];
    corrs_by_cl = cell(tot_cl, 1);
    cluster_acceptance_threshold = 0.9;
    
    for count_cl=1:tot_cl
        all_imgs_rgb = all_imgs_rgb_by_cl{count_cl};
        corrs = zeros(size(all_imgs_rgb, 1));
        for ii1=1:size(all_imgs_rgb, 1)
            tmp_img1 = uint8(squeeze(all_imgs_rgb(ii1, :, :, :)));
            tmp_img1 = rgb2gray(tmp_img1);
            for ii2=1:size(all_imgs_rgb, 1)
                tmp_img2 = uint8(squeeze(all_imgs_rgb(ii2, :, :, :)));
                tmp_img2 = rgb2gray(tmp_img2);
                %SSIM between each pair of images in each cluster
                corrs(ii1, ii2) = ssim(tmp_img1, tmp_img2);
            end
        end
        corrs2 = corrs;
        corrs2(1:size(all_imgs_rgb, 1)+1:end) = nan;
        mediana = nanmedian(corrs2(:));
        media = nanmean(corrs2(:));
        sqm = nanstd(corrs2(:));
        mediane = [mediane mediana];
        medie = [medie media];
        
        corrs_by_cl{count_cl}.corrs = corrs;
        corrs_by_cl{count_cl}.corrs2 = corrs2;
    end
    %median measure of quality
    mediana2 = median(mediane);
    

%% evaluate clusters
    %based on the median computed
    cluster_eval = cell(tot_cl, 1);
    for count_cl=1:tot_cl
        
        all_imgs_rgb = all_imgs_rgb_by_cl{count_cl};
        corrs = corrs_by_cl{count_cl}.corrs;
        corrs2 = corrs_by_cl{count_cl}.corrs2;
        mediana = nanmedian(corrs2(:));
        media = nanmean(corrs2(:));
        sqm = nanstd(corrs2(:));
        
        [~, sorted_ii_inds] = sort(sum(corrs < mediana2, 2));
        
        cluster_eval{count_cl}.samples_scores = zeros(size(all_imgs_rgb, 1), 1);
        cluster_eval{count_cl}.samples_oks = false(size(all_imgs_rgb, 1), 1);
        for ii1=1:size(all_imgs_rgb, 1)
            ii = sorted_ii_inds(ii1);
            
            cluster_eval{count_cl}.samples_scores(ii) = sum((corrs(ii, :) < mediana2), 2) / size(all_imgs_rgb, 1);
            cluster_eval{count_cl}.samples_oks(ii) = sum((corrs(ii, :) < mediana2), 2) / size(all_imgs_rgb, 1) < 0.6;
        end
        
        
        cluster_eval{count_cl}.score = mediana / mediana2;
        cluster_eval{count_cl}.ok = true;
        if mediana < cluster_acceptance_threshold * mediana2
            cluster_eval{count_cl}.ok = false;
        end
        close all
    end

%% clean clusters based on evaluation
    %(produces visual summaries)
    for count_cl=1:tot_cl
        if ~cluster_eval{count_cl}.ok
            continue
        end
        if size(contexts_and_convexhulls_by_cl{count_cl}, 1) == 0
            continue
        end
        
        all_imgs_rgb = all_imgs_rgb_by_cl{count_cl};
        good_cl_inds = find(cluster_eval{count_cl}.samples_oks );
        
        n_subplots = ceil(sqrt(numel(good_cl_inds)));
        close all
        
        f = figure;
        f.Visible = 'off';
        
        %for each image in cluster (after filtering out bad ones based on SSIM)
        for ii=1:numel(good_cl_inds)
%             subplot(n_subplots, n_subplots, ii);
            gci = good_cl_inds(ii);
            
            tmp_mask2_properties = contexts_and_convexhulls_by_cl{count_cl}.maskprops{gci};
            tmp_props2 = contexts_and_convexhulls_by_cl{count_cl}.bboxes{gci};
            
            
            tmp_occlusion_mask = contexts_and_convexhulls_by_cl{count_cl}.conveximgs{gci};
            %for each region on the mask
            for cci=1:size(tmp_mask2_properties, 1)
                tmp_empty_mask = zeros(size(contexts_and_convexhulls_by_cl{count_cl}.contexts{gci}, 1), size(contexts_and_convexhulls_by_cl{count_cl}.contexts{gci}, 2));
                tmp_bbox_xyxy = [round(tmp_mask2_properties(cci).BoundingBox(1:2)) round(tmp_mask2_properties(cci).BoundingBox(1:2) + tmp_mask2_properties(cci).BoundingBox(3:4)) - [1,1] ];
                tmp_empty_mask(tmp_bbox_xyxy(2):tmp_bbox_xyxy(4), tmp_bbox_xyxy(1):tmp_bbox_xyxy(3)) = tmp_mask2_properties(cci).Image;
                
                tmp_rp_pixels = zeros(size(contexts_and_convexhulls_by_cl{count_cl}.contexts{gci}, 1), size(contexts_and_convexhulls_by_cl{count_cl}.contexts{gci}, 2));
                tmp_rp_pixels(tmp_props2(2):tmp_props2(4), tmp_props2(1):tmp_props2(3)) = 1;
                tmp_rp_inters_pixels = tmp_rp_pixels & tmp_empty_mask;
                
                if sum(tmp_rp_inters_pixels(:)) > 0
                    tmp_occlusion_mask = tmp_occlusion_mask | tmp_empty_mask;
                end
            end
            tmp_img2 = contexts_and_convexhulls_by_cl{count_cl}.contexts{gci};
            %grey out outside of mask regions
            tmp_img2(~repmat(tmp_occlusion_mask, 1, 1, 3)) = 64;
            imshow(tmp_img2, 'Border', 'tight')
            %draw red region proposal convex hull on image
            hold on
            line(contexts_and_convexhulls_by_cl{count_cl}.chrs{gci}, contexts_and_convexhulls_by_cl{count_cl}.chcs{gci}, 'Color', 'r', 'LineWidth',3);
            hold off
            
            %draw mask regions
            for cci=1:size(tmp_mask2_properties, 1)
                tmp_empty_mask = zeros(size(contexts_and_convexhulls_by_cl{count_cl}.contexts{gci}, 1), size(contexts_and_convexhulls_by_cl{count_cl}.contexts{gci}, 2));
                tmp_bbox_xyxy = [round(tmp_mask2_properties(cci).BoundingBox(1:2)) round(tmp_mask2_properties(cci).BoundingBox(1:2) + tmp_mask2_properties(cci).BoundingBox(3:4)) - [1,1] ];
                tmp_empty_mask(tmp_bbox_xyxy(2):tmp_bbox_xyxy(4), tmp_bbox_xyxy(1):tmp_bbox_xyxy(3)) = tmp_mask2_properties(cci).Image;
                
                tmp_rp_pixels = zeros(size(contexts_and_convexhulls_by_cl{count_cl}.contexts{gci}, 1), size(contexts_and_convexhulls_by_cl{count_cl}.contexts{gci}, 2));
                tmp_rp_pixels(tmp_props2(2):tmp_props2(4), tmp_props2(1):tmp_props2(3)) = 1;
                tmp_rp_inters_pixels = tmp_rp_pixels & tmp_empty_mask;
                
                if sum(tmp_rp_inters_pixels(:)) > 0
                    % if overlaps with region proposal
                    tmp_boundary = bwboundaries(tmp_empty_mask);
                    for bbi=1:numel(tmp_boundary)
                        tb = tmp_boundary{bbi};
                        hold on
                        plot(tb(:, 2), tb(:, 1), 'Color', 'c', 'LineWidth',2)
                        hold off
                    end
                end
            end
            print(['police_van_poster/' num2str(count_cl) '_' num2str(ii)],'-dpng', '-r0');
            
        end
        print([out_folder '/cleaned_usertest_contexts+maskregions_' num2str(count_cl)],'-dpng', '-r0');
        close all
    end
    
    
    'figure creation finished, saving mat file...'
    save([out_base_folder class_folders(class_ind).name '.mat'], '-v7.3');
    toc(tstart)
end

