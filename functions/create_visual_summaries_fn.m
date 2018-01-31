function create_visual_summaries_fn(cluster_eval,tot_cl,contexts_and_convexhulls_by_cl, out_folder)
%CREATE_VISUAL_SUMMARIES_FN Creates the visual summaries and saves them to
%disk
for count_cl=1:tot_cl
    if ~cluster_eval{count_cl}.ok
        continue
    end
    if size(contexts_and_convexhulls_by_cl{count_cl}, 1) == 0
        continue
    end

    good_cl_inds = find(cluster_eval{count_cl}.samples_oks );

    n_subplots = ceil(sqrt(numel(good_cl_inds)));
    close all

    f = figure;
    f.Visible = 'off';
    f.PaperUnits = 'inches';
    f.PaperPosition = [0 0 10 10];

    %for each image in cluster (after filtering out bad ones based on SSIM)
    for ii=1:numel(good_cl_inds)
        subplot(n_subplots, n_subplots, ii);
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

    end
    print([out_folder '/cleaned_usertest_contexts+maskregions_' num2str(count_cl)],'-dpng', '-r0');
    close all
end
end

