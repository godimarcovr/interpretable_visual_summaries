function [tmp_good_inds] = prune_rps_redundant_fn(tmp_good_inds,tmp_props, similarity_mat_part, same_region_thresh, scale_factor_thresh )
%PRUNE_RPS_REDUNDANT_FN given a list of region proposals, remove in a greedy way those
%which are redundant in terms of position and scale

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

                [tmp_sval, ~] = sort(similarity_mat_part(good_ind2, :),'descend'); %heuristic, keep those with better scores
                best_scores = [best_scores mean(tmp_sval(1:100))]; 
            else
                to_keep = [to_keep good_ind2];
            end
        end
    end
    [~, best_ind] = max(best_scores);
    tmp_good_inds = [redundant_inds(best_ind) to_keep];
end
end

