function [cluster_eval] = evaluate_clusters_fn(all_imgs_rgb_by_cl,tot_cl, corrs_by_cl, mediana2,cluster_acceptance_threshold)
%EVALUATE_CLUSTERS_FN Based on similarity measures, decide whether to
%discard a cluster or not


%based on the median computed
cluster_eval = cell(tot_cl, 1);
for count_cl=1:tot_cl

    all_imgs_rgb = all_imgs_rgb_by_cl{count_cl};
    corrs = corrs_by_cl{count_cl}.corrs;
    corrs2 = corrs_by_cl{count_cl}.corrs2;
    mediana = nanmedian(corrs2(:));
%     media = nanmean(corrs2(:));
%     sqm = nanstd(corrs2(:));

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
end
end

