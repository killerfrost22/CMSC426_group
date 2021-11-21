function [WarpedFrame, WarpedMask, WarpedMaskOutline, WarpedLocalWindows] = calculateGlobalAffine(IMG1,IMG2,Mask,MaskOutline,Windows)
% CALCULATEGLOBALAFFINE: finds affine transform between two frames, and applies it to frame1, the mask, and local windows.

% Detect SIFT features and match them
gray1 = single(rgb2gray(IMG1));
gray2 = single(rgb2gray(IMG2));
[f1,d1] = vl_sift(gray1);
[f2,d2] = vl_sift(gray2);
matches = vl_ubcmatch(d1,d2);

% Calculate affine transform
matchedPoints1 = f1(1:2,matches(1,:))';
matchedPoints2 = f2(1:2,matches(2,:))';
tform = estimateGeometricTransform(matchedPoints1,matchedPoints2,'affine');

% Warp stuff
outputView = imref2d(size(gray2));
WarpedFrame = imwarp(IMG1,tform,'OutputView',outputView);
WarpedMask = imwarp(Mask,tform,'OutputView',outputView);
%WarpedMaskOutline = imwarp(MaskOutline,tform,'OutputView',outputView);
WarpedMaskOutline = bwperim(WarpedMask,4);
WarpedLocalWindows = transformPointsForward(tform,Windows);

end