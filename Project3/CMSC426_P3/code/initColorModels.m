function ColorModels = initColorModels(IMG, Mask, MaskOutline, LocalWindows, BoundaryWidth, WindowWidth)
% INITIALIZAECOLORMODELS Initialize color models.  ColorModels is a struct you should define yourself.
%
% Must define a field ColorModels.Confidences: a cell array of the color confidence map for each local window.
lab = rgb2lab(IMG);
MaskOutline_b = imdilate(MaskOutline, strel('disk',BoundaryWidth));
num_GMM_components = 2;
numLocalWindows = size(LocalWindows, 1);
% Choose color space
img_lab = rgb2lab(IMG);

ColorModels.D = cell(size(LocalWindows,1),1);
ColorModels.ForeModels = cell(size(LocalWindows,1),1);
ColorModels.ForePixels = cell(size(LocalWindows,1),1);
ColorModels.BackModels = cell(size(LocalWindows,1),1);
ColorModels.BackPixels = cell(size(LocalWindows,1),1);
ColorModels.Confidences = cell(size(LocalWindows,1),1);

for i = 1:numLocalWindows
    x = LocalWindows(i,1);
    y = LocalWindows(i,2);
    yRange = (y-(WindowWidth/2)):(y+(WindowWidth/2 - 1));
    xRange = (x-(WindowWidth/2)):(x+(WindowWidth/2 - 1));
    img_window = img_lab(yRange,xRange,:);
    mask_window = Mask(yRange,xRange);
    mask_outline_window = MaskOutline(yRange,xRange);
    
    % Create a mask to exclude points too close to boundary
    D = bwdist(mask_outline_window);
    ColorModels.D{i} = D;
    valid_pixel_mask = zeros(size(D));
    valid_pixel_mask(find(D > BoundaryWidth)) = 1;
    foreground_mask = mask_window & valid_pixel_mask;
    background_mask = ~mask_window & valid_pixel_mask;

    foreground = img_window.*repmat(foreground_mask,[1 1 3]);
    background = img_window.*repmat(background_mask,[1 1 3]);
    foreground_pixels = reshape(foreground,[],3);
    foreground_pixels = foreground_pixels(any(foreground_pixels,2),:);
    if size(foreground_pixels,1) < 3
        imshow(lab2rgb(img_window))
        figure()
        imshow(mask_window)
    end
    background_pixels = reshape(background,[],3);
    background_pixels = background_pixels(any(background_pixels,2),:);
    if size(background_pixels,1) < 3
        imshow(lab2rgb(img_window))
        figure()
        imshow(mask_window)
    end
    ColorModels.ForePixels{i} = foreground_pixels;
    ColorModels.BackPixels{i} = background_pixels;
    ColorModels.ForeModels{i} = fitgmdist(foreground_pixels, num_GMM_components, 'RegularizationValue', 0.001, 'Options', statset('MaxIter', 100));
    ColorModels.BackModels{i} = fitgmdist(background_pixels, num_GMM_components, 'RegularizationValue', 0.001, 'Options', statset('MaxIter', 100));
    
    % Compute confidence 
    pixels = reshape(img_window,[],3);
    p_c_F = pdf(ColorModels.ForeModels{i}, pixels);
    p_c_B = pdf(ColorModels.BackModels{i}, pixels);
    p_c = p_c_F./(p_c_F + p_c_B);
    p_c(isnan(p_c)) = 0;
    integrand = abs(reshape(double(mask_window),[],1)-p_c);
    sigma_c = WindowWidth/2;
    w_c = exp(-D.^2/(sigma_c^2));
    w_c = reshape(w_c,[],1);
    ColorModels.Confidences{i} = 1 - sum(integrand.*w_c)/sum(w_c);
end
end

