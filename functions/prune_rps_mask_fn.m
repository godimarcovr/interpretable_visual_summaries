function [tmp_good_inds] = prune_rps_mask_fn(sharp_mask_path, tmp_props, valid_rp_thresh,num_op)
%PRUNE_RPS_MASK_FN given a mask image (one channel, same size as image) and
%a list of rectangular region proposals, return the indexes of the ones
%with the correct overlap
% read mask
mask = imread(fullfile(sharp_mask_path.folder, sharp_mask_path.name));
mask = mask > 128;
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
end

