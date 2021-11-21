function [mask, LocalWindows, ColorModels, ShapeConfidences] = ...
    updateModels(...
        NewLocalWindows, ...
        LocalWindows, ...
        CurrentFrame, ...
        warpedMask, ...
        warpedMaskOutline, ...
        WindowWidth, ...
        ColorModels, ...
        ShapeConfidences, ...
        ProbMaskThreshold, ...
        fcutoff, ...
        SigmaMin, ...
        R, ...
        A ...
    )

fprintf('Merging Local Windows\n')
[mask, ShapeConfidences] = mergeLocalWindows(CurrentFrame, ColorModels, ShapeConfidences, warpedMask, NewLocalWindows, WindowWidth);
LocalWindows = NewLocalWindows;

numLocalWindows = size(NewLocalWindows, 1);
confidence = cell(numLocalWindows, 1);
foreground = cell(numLocalWindows, 1);
background = cell(numLocalWindows, 1);
f_prob = cell(numLocalWindows, 1);
b_prob = cell(numLocalWindows, 1);
distances = cell(numLocalWindows, 1);
probs = cell(numLocalWindows, 1);
w = cell(numLocalWindows, 1);

end

% Use old color model with new foreground mask and new local windows to
% update color confidences for each new local window
function ColorModels = updateColorConfidences(ColorModels, img, warpedMask, warpedMaskOutline, NewLocalWindows, WindowWidth)
img_lab = rgb2lab(img);
for i = 1:size(NewLocalWindows,1)
    x = NewLocalWindows(i,1);
    y = NewLocalWindows(i,2);
    if x-WindowWidth/2 < 1 || x+(WindowWidth/2-1) > size(img_lab,2) || y-WindowWidth/2 < 1 || y+(WindowWidth/2-1) > size(img_lab,1)
        continue
    end
    yRange = (y-(WindowWidth/2)):(y+(WindowWidth/2 - 1));
    xRange = (x-(WindowWidth/2)):(x+(WindowWidth/2 - 1));
    img_window = img_lab(yRange,xRange,:);
    mask_window = warpedMask(yRange,xRange);
    mask_outline_window = warpedMaskOutline(yRange,xRange);
    D = bwdist(mask_outline_window);
    ColorModels.D{i} = D;
    pixels = reshape(img_window,[],3);
    p_c_F = pdf(ColorModels.ForeModels{i}, pixels);
    p_c_B = pdf(ColorModels.BackModels{i}, pixels);
    p_c = p_c_F./(p_c_F + p_c_B);
    integrand = abs(reshape(double(mask_window),[],1)-p_c);
    sigma_c = WindowWidth/2;
    w_c = exp(-D.^2/(sigma_c^2));
    w_c = reshape(w_c,[],1);
    ColorModels.Confidences{i} = 1 - sum(integrand.*w_c)/sum(w_c);
end
end

function ShapeConfidences = updateShapeConfidences(ShapeConfidences, ColorModels, WindowWidth, SigmaMin, A, fcutoff, R)
for i = 1:size(ShapeConfidences.Confidences,1)
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

function ColorModels = updateColorModels(ColorModels, ShapeConfidences, img, warpedMask, NewLocalWindows, WindowWidth, ProbMaskThreshold)

high_threshold = ProbMaskThreshold;
low_threshold = 1-high_threshold;
foreground_increase_limit = 1.25;
num_GMM_components = 3;
img_lab = rgb2lab(img);

for i = 1:size(NewLocalWindows,1)
    x = NewLocalWindows(i,1);
    y = NewLocalWindows(i,2);
    if x-WindowWidth/2 < 1 || x+(WindowWidth/2-1) > size(img_lab,2) || y-WindowWidth/2 < 1 || y+(WindowWidth/2-1) > size(img_lab,1)
        continue
    end
    yRange = (y-(WindowWidth/2)):(y+(WindowWidth/2 - 1));
    xRange = (x-(WindowWidth/2)):(x+(WindowWidth/2 - 1));
    img_window = img_lab(yRange,xRange,:);
    mask_window = warpedMask(yRange,xRange);
    foreground_mask = zeros(WindowWidth);
    background_mask = zeros(WindowWidth);
    foreground_mask(find(ShapeConfidences.Confidences{i} > high_threshold)) = 1;
    background_mask(find(ShapeConfidences.Confidences{i} < low_threshold)) = 1;
    foreground = img_window.*repmat(foreground_mask,[1 1 3]);
    background = img_window.*repmat(background_mask,[1 1 3]);
    foreground_pixels = reshape(foreground,[],3);
    foreground_pixels = foreground_pixels(any(foreground_pixels,2),:);
    if size(foreground_pixels,1) < 3
        foreground_pixels = cat(1,ColorModels.ForePixels{i},foreground_pixels);
    end
    background_pixels = reshape(background,[],3);
    background_pixels = background_pixels(any(background_pixels,2),:);
    if size(background_pixels,1) < 3
        background_pixels = cat(1,ColorModels.BackPixels{i},background_pixels);
    end
    NewForeModel = fitgmdist(foreground_pixels, num_GMM_components, 'RegularizationValue', 0.001, 'Options', statset('MaxIter', 50));
    NewBackModel = fitgmdist(background_pixels, num_GMM_components, 'RegularizationValue', 0.001, 'Options', statset('MaxIter', 50));
    
    % Test updated color model for sampling errer
    pixels = reshape(img_window,[],3);
    p_c_F = pdf(ColorModels.ForeModels{i}, pixels);
    p_c_B = pdf(ColorModels.BackModels{i}, pixels);
    p_c_h = p_c_F./(p_c_F + p_c_B);
    p_c_h(isnan(p_c_h)) = 0;
    p_c_F = pdf(NewForeModel, pixels);
    p_c_B = pdf(NewBackModel, pixels);
    p_c_u = p_c_F./(p_c_F + p_c_B);
    p_c_u(isnan(p_c_u)) = 0;
    foreground_h = logical(round(p_c_h));
    foreground_u = logical(round(p_c_u));
    num_fore_pixels_h = sum(foreground_h,'all');
    num_fore_pixels_u = sum(foreground_u,'all');
    if num_fore_pixels_u < foreground_increase_limit * num_fore_pixels_h
        ColorModels.ForePixels{i} = foreground_pixels;
        ColorModels.BackPixels{i} = background_pixels;
        ColorModels.ForeModels{i} = NewForeModel;
        ColorModels.BackModels{i} = NewBackModel;
        integrand = abs(reshape(double(mask_window),[],1)-p_c_u);
        sigma_c = WindowWidth/2;
        D = ColorModels.D{i};
        w_c = exp(-D.^2/(sigma_c^2));
        w_c = reshape(w_c,[],1);
        ColorModels.Confidences{i} = 1 - sum(integrand.*w_c)/sum(w_c);
    end
end

end

function [mask,ShapeConfidences] = mergeLocalWindows(img, ColorModels, ShapeConfidences, warpedMask, NewLocalWindows, WindowWidth)

epsilon = 0.1;
foreground_threshold = 0.5;
background_threshold = 0.1;
img_lab = rgb2lab(img);

warpedMaskBoundary = imdilate(bwperim(warpedMask,4),strel('disk', 5, 4));
warpedMaskInterior = logical(warpedMask) - logical(warpedMaskBoundary);
warpedMaskInterior(warpedMaskInterior>1) = 1;
warpedMaskInterior(warpedMaskInterior<0) = 0;

%total_numerator = 0.05*double(ShapeConfidences.prob_mask) + 0.8*double(warpedMaskInterior);
%total_numerator = double(warpedMaskInterior);
total_numerator = zeros(size(warpedMask), 'double');

total_denominator = zeros(size(warpedMask), 'double');
p_F = cell(size(NewLocalWindows,1),1);
BW = zeros(WindowWidth);
BW(WindowWidth/2+1,WindowWidth/2+1) = 1;
denominator = double(ones(WindowWidth,'double')./(bwdist(BW) + epsilon)); % (|x-c_k| + eps) will be the same for every local window
for k = 1:size(NewLocalWindows,1)
    x = NewLocalWindows(k,1);
    y = NewLocalWindows(k,2);
    if x-WindowWidth/2 < 1 || x+(WindowWidth/2-1) > size(img_lab,2) || y-WindowWidth/2 < 1 || y+(WindowWidth/2-1) > size(img_lab,1)
        continue
    end
    yRange = (y-(WindowWidth/2)):(y+(WindowWidth/2 - 1));
    xRange = (x-(WindowWidth/2)):(x+(WindowWidth/2 - 1));
    img_window = img_lab(yRange,xRange,:);
    mask_window = warpedMask(yRange,xRange);
    f_s = ShapeConfidences.Confidences{k};
    pixels = reshape(img_window,[],3);
    p_c_F = pdf(ColorModels.ForeModels{k}, pixels);
    p_c_B = pdf(ColorModels.BackModels{k}, pixels);
    p_c = p_c_F./(p_c_F + p_c_B);
    p_c(isnan(p_c)) = 0;
    p_c = reshape(p_c,size(f_s));
    p_f{k} = f_s.*mask_window + (1 - f_s).*p_c;
    
    % Window merging
    numerator = p_f{k}.*(denominator);
    total_numerator(yRange,xRange) = total_numerator(yRange,xRange) + numerator;
    total_denominator(yRange,xRange) = total_denominator(yRange,xRange) + denominator;
end
total_denominator(~total_denominator) = 1;
total_probability_map = total_numerator./total_denominator;
total_probability_map = total_probability_map + warpedMaskInterior;
total_probability_map(total_probability_map>1) = 1;
total_probability_map(total_probability_map<0) = 0;
ShapeConfidences.prob_mask = total_probability_map;
%figure();
%imshow(total_probability_map);
foreground_mask = total_probability_map > foreground_threshold;
background_mask = total_probability_map < background_threshold;

%{
figure('Name', 'Foreground Mask');
imshow(foreground_mask);
figure('Name', 'Background Mask');
imshow(background_mask);
%}

% Lazysnapping
%L = superpixels(img,500);
%mask = lazysnapping(img,L,foreground_mask,background_mask);

mask = foreground_mask;
end
