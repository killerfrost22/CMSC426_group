function ShapeConfidences = initShapeConfidences(LocalWindows, ColorConfidences, WindowWidth, SigmaMin, A, fcutoff, R)
% INITSHAPECONFIDENCES Initialize shape confidences.  ShapeConfidences is a struct you should define yourself.

ShapeConfidences.Confidences = cell(size(LocalWindows,1),1);
for i=1:size(LocalWindows,1)
    sigma_s = SigmaMin;
    f_c = ColorModels.Confidences{i};
    if fcutoff < f_c && f_c <= 1
        sigma_s = SigmaMin + A*(f_c - fcutoff)^R;
    end
    D = ColorModels.D{i};
    f_s = ones(WindowWidth) - exp(-D.^2./(sigma_s^2));
    ShapeConfidences.Confidences{i} = f_s;
end
end
