function warped_img2= dense_warp_fn(img1, proposals1, img2, proposals2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


viewA = load_view(imgA,opA,featA);
viewB = load_view(imgB,opB,featB);

bPost=true; % applying post processing using SDFilering
match = flow_field_generation(viewA, viewB, confidence, sdf, bPost);
end

