function ShapeConfidences = initShapeConfidences(LocalWindows, ColorModels, WindowWidth, SigmaMin, A, fcutoff, R)
% INITSHAPECONFIDENCES Initialize shape confidences.  ShapeConfidences is a struct you should define yourself.
numLocalWindows = size(LocalWindows, 1);
ShapeConfidences.Confidences = cell(size(LocalWindows,1),1);
% f_s = cell(numLocalWindows);
for i=1:numLocalWindows
    sigma_s = SigmaMin;
    f_c = ColorModels.Confidences{i};
    if (fcutoff < f_c && f_c <= 1) 
        sigma_s = SigmaMin + A*(f_c - fcutoff)^R;
    else
        sigma_s = SigmaMin;
    end
    D = ColorModels.D{i};
    f_s = ones(WindowWidth) - exp(-D.^2./(sigma_s^2));
    ShapeConfidences.Confidences{i} = f_s;
end
end
