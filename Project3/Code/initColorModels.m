function ColorModels = initializeColorModels(IMG, Mask, MaskOutline, LocalWindows, BoundaryWidth, WindowWidth)
% INITIALIZAECOLORMODELS Initialize color models.  ColorModels is a struct you should define yourself.
% This is a comment
% This is another comment
% Must define a field ColorModels.Confidences: a cell array of the color confidence map for each local window.

%% A GMM model
ColorModels = struct('Models', {{}}, 'fgPixels', {{}}, 'bgPixels', {{}}, 'f_gmms', {{}}, 'b_gmms', {{}}, 'Confidences', {{}}, 'Distances', {{}});
[height, width, colors] = size(IMG);
sigma_c = WindowWidth / 2;
IMG = rgb2lab(IMG);

for i = 1:size(LocalWindows, 1) % iterate through each local window, where the row represents (x, y) of center
    % create a GMM model with 3 gaussians
    % data: RGB val of pixels just in foreground -> generates model for
    % probability that a pixel is in the foreground (take pixels that are
    % white in the mask, and dist > 5 from MaskOutline)
    c = LocalWindows(i, 1);
    r = LocalWindows(i, 2);
    top = r - WindowWidth/2;
    bottom = r + WindowWidth/2;
    left = c - WindowWidth/2;
    right = c + WindowWidth/2;
    if (top < 0)
        top = 0;
    end
    if (bottom > height)
        bottom = height;
    end
    if (left < 0)
        left = 0;
    end
    if (right > width)
        right = width;
    end
    
    window = IMG(top:bottom, left:right, :);
    window_outline = MaskOutline(top:bottom, left:right, :);
    mask_window = Mask(top:bottom, left:right, :);
    %imwrite(mask_window, 'shape_model_init.jpg')
    D = bwdist(window_outline); % stores dist for each pixel from the boundary
    foreground_data = [];
    background_data = [];
    for row = 1:size(window, 1)
        for col = 1:size(window, 2)
            % if the distance from the pixel to the boundary is >
            % BoundaryWidth, and the pixel is in the foreground:
            if(D(row, col) > BoundaryWidth)
                r = window(row, col, 1);
                g = window(row, col, 2);
                b = window(row, col, 3);
                color_array = [r g b];
                if (mask_window(row, col) == 1)
                    foreground_data = [ foreground_data ; color_array];
                else
                    background_data = [ background_data ; color_array];
                end
            end
        end
    end
    % fit GMM with 2 gaussians for foreground, and 1 gaussian for
    % background:
    f_gmm = fitgmdist(foreground_data,1);
    b_gmm = fitgmdist(background_data,1);
    model = @(X) pdf(f_gmm, X) ./ (pdf(f_gmm, X) + pdf(b_gmm, X));
    
    % show model applied to the window:
    num = 0.0;
    den = 0.0;
    img_colormodel = zeros(size(window, 1), size(window, 2));
    for row = 1:size(window, 1)
        for col = 1:size(window, 2)
            r = window(row, col, 1);
            g = window(row, col, 2);
            b = window(row, col, 3);
            color_array = [r g b];
            weight = exp(-D(row, col)^2/sigma_c^2);
            num = num + abs(mask_window(row, col) - model(color_array)) *  weight;
            den = den + weight;
            img_colormodel(row, col) = model(color_array);
        end
    end
    
    %imwrite(img_colormodel, 'color_model_init.jpg');
    confidence = 1 - num/den;
    ColorModels.fgPixels{i} = foreground_data;
    ColorModels.bgPixels{i} = background_data;
    ColorModels.f_gmms{i} = f_gmm;
    ColorModels.b_gmms{i} = b_gmm;
    ColorModels.Confidences{i} = confidence;
    ColorModels.Distances{i} = D;
    
end

ColorModels.Confidences

end

