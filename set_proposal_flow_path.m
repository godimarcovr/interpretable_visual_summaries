set_proposal_flow_folder; %set path in set_proposal_flow_folder script!

%% set paths
% object proposals
addpath(fullfile(proposal_flow_folder_path,'object-proposal'));
% selective search
addpath(fullfile(proposal_flow_folder_path,'object-proposal/SelectiveSearchCodeIJCV'));
addpath(fullfile(proposal_flow_folder_path,'object-proposal/SelectiveSearchCodeIJCV/Dependencies'));
% common functions
addpath(fullfile(proposal_flow_folder_path,'commonFunctions'));
% feature
addpath(genpath(fullfile(proposal_flow_folder_path, 'feature')));
% matching algorithm
addpath(fullfile(proposal_flow_folder_path,'/algorithms'));
% dense correspondence
addpath(fullfile(proposal_flow_folder_path,'/denseCorrespondence'));

% SD filter
addpath(fullfile(proposal_flow_folder_path,'/sdFilter'));
% Dense Warp path
addpath(fullfile(proposal_flow_folder_path,'_demo-DenseFlow/'));

addpath(fullfile(pwd,'functions/'));