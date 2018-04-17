function [match, weight, confidence, warp] = compute_flow_fn(img1_path, img2_path,featA,featB, num_op, proposals, doWarp)
% demo code for computing dense flow field
% using ProposalFlow (LOM+SS)
% show object proposal matching
bShowMatch = false;
set_conf;
% % show dense flow field
% bShowFlow = true;

%set_conf;

% num_op=500; %number of object proposals

% fprintf(' + Parsing images\n\n');
% imgA = imread(fullfile(conf.datasetDir,'Cars_008a.png'));
% imgB = imread(fullfile(conf.datasetDir,'Cars_014b.png'));
imgA = imread(img1_path);
imgB = imread(img2_path);

if num_op == 0
    proposalA = [1 1 size(imgA, 2) size(imgA, 1) ];
    proposalB = [1 1 size(imgB, 2) size(imgB, 1) ];
else
    if size(proposals, 1) == 0
        [proposalA, ~] = SS(imgA, num_op);% (x,y) coordinates ([col,row]) for left-top and right-bottom points
        [proposalB, ~] = SS(imgB, num_op);% (x,y) coordinates ([col,row]) for left-top and right-bottom points
    else
        if iscell(proposals)
            proposalA = proposals{1};
            proposalB = proposals{2};
        else
            proposalA = proposals(1:num_op, :);
            proposalB = proposals(num_op + 1: end, :);
        end
    end
end
opA.coords=proposalA;
opB.coords=proposalB;
clear proposalA; clear proposalB;

if num_op == 0
    viewA = load_view(imgA,opA,featA, 'cand', [1]);
    viewB = load_view(imgB,opB,featB, 'cand', [1]);
else
    viewA = load_view(imgA,opA,featA);
    viewB = load_view(imgB,opB,featB);
end
% imgB = imgB(:,end:-1:1, :);

ishog = true;


% options for matching
opt.feature = 'HOG';
opt.bDeleteByAspect = true;
opt.bDensityAware = false;
opt.bSimVote = true;
opt.bVoteExp = true;


% matching algorithm
% NAM: naive appearance matching
% PHM: probabilistic Hough matching
% LOM: local offset matching
tic;
confidence = feval( @LOM, viewA, viewB, opt );

% %%mod
% if flip
%     res=reshape(viewB.desc,[8,8,31,num_op]);
%     res=res(:,8:-1:1,vl_hog('permutation'),:);
%     res=res(:);
%     viewBi=viewB;
%     viewBi.desc=reshape(res,[8*8*31,num_op]);
%     confidence2 = feval( @LOM, viewA, viewBi, opt );
%     confidence=max(confidence,confidence2);
% end
%%
% fprintf('   - %s took %.2f secs.\n\n', func2str(@LOM), toc);
t1=toc;

% ===============================================================
% show object proposal matching
% ===============================================================
[confidenceA, max_id ] = max(confidence,[],2);
match = [ 1:numel(max_id); max_id'];
if bShowMatch
    
    hFig_match = figure(1); clf;
    imgInput = appendimages( viewA.img, viewB.img, 'h' );
    imshow(rgb2gray(imgInput)); hold on;
    showColoredMatches(viewA.frame, viewB.frame, match,...
        confidenceA, 'offset', [ size(viewA.img,2) 0 ], 'mode', 'box');
end

weight = confidenceA;
[ weight, idxC ] = sort(weight, 'descend');
match = match(:, idxC);

bPost=true; % applying post processing using SDFilering
if doWarp
    warp = flow_field_generation(viewA, viewB, confidence, sdf, bPost);
else
    warp = [];
end

end
% % ===============================================================
% % computing dense flow field
% % ===============================================================
% % fprintf(' + Computing dense correspondnece ');
% tic;
% 
% % fprintf('took %.2f secs.\n\n',toc);
% t2=toc;
% 
% % fprintf('==================================\n');
% % fprintf('Total flow took %.2f secs\n',t1+t2);
% % fprintf('==================================\n');
% 
% save(fullfile(conf.resultDir,'flow.mat'), 'match');
% 
% 
% if bShowFlow
%     clf(figure(2),'reset');
%     imgInput = appendimages( viewA.img, viewB.img, 'h' );
%     figure(2);imshow(imgInput);hold on;
%     figure(3);imshow(flowToColor(cat(3,match.vx,match.vy)));%Flow =cat(3,match.vx,match.vy);cquiver(Flow(1:10:end,1:10:end,:));
%     WarpedImg=warpImage(im2double(viewB.img),match.vx,match.vy);
%     figure(4);imshow(WarpedImg);
% end
