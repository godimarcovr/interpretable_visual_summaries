function [feat] = compute_hog_descriptor_fn(img_path, num_op, proposals)
% demo code for computing dense flow field
% using ProposalFlow (LOM+SS)
% show object proposal matching
bShowMatch = false;

% % show dense flow field
% bShowFlow = true;

%set_conf;

% num_op=500; %number of object proposals

% fprintf(' + Parsing images\n\n');
% img = imread(fullfile(conf.datasetDir,'Cars_008a.png'));
% imgB = imread(fullfile(conf.datasetDir,'Cars_014b.png'));
img = imread(img_path);
% imgB = imgB(:,end:-1:1, :);

ishog = true;

% ===============================================================
% extracting object proposals using SelectiveSearch
% ===============================================================
% fprintf(' + Extrating object proposals ');
tic;
if num_op == 0
    proposal = [1 1 size(img, 2) size(img, 1) ];
else
    if size(proposals, 1) == 0
        [proposal, ~] = SS(img, num_op);% (x,y) coordinates ([col,row]) for left-top and right-bottom points
    else
        if iscell(proposals)
            proposal = proposals{1};
        else
            proposal = proposals(1:num_op, :);
        end
    end
end
op.coords=proposal;
clear proposal;
% fprintf('took %.2f secs.\n\n',toc);

% ===============================================================
% extrating feature descriptors
% ===============================================================
% fprintf(' + Extrating featrues ');
tic;
if ishog
    feat =  extract_segfeat_hog(img,op);
    opt.feature = 'HOG';
end

end

