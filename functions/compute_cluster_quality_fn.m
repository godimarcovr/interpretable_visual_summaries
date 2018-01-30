function [corrs_by_cl,mediana2] = compute_cluster_quality_fn(all_imgs_rgb_by_cl,tot_cl)
%COMPUTE_CLUSTER_QUALITY_FN Computes measures of similarity between cluster
%elements to evaluate whether to eliminate them or not
mediane = [];
medie = [];
corrs_by_cl = cell(tot_cl, 1);

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
end

