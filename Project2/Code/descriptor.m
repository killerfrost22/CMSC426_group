function [descriptors] = feature_descriptors(img_grayscale, x_best, y_best)
    [~, n_best] = size(x_best);
    [y_size, x_size] = size(img_grayscale);
    descriptors = [];
    for i = 1:n_best
        % Patch is of size 41x41, so point is the actual center
        x = x_best(i);
        y = y_best(i);
        % TODO?
        if or(or(or(x <= 21, y<=21), x + 21> x_size), y + 21 > y_size)
            v = zeros(8*8,1);
            descriptors = [descriptors v];
            continue;
        end
        patch = img_grayscale(y-21:y+21, x-21:x+21);
        % Gaussian blur
        sig = 10;
        blurred = imgaussfilt(patch, sig);
        % Resize
        resized = imresize(blurred, [8 8]);
        % Reshape
        reshaped = double(reshape(resized, [64, 1]));
        % Now we need to standardize
        std_dev = std(reshaped);
        mean_reshaped = mean(reshaped);
        standardized = reshaped - mean_reshaped;
        standardized = standardized / std_dev;
        descriptors = [descriptors standardized];
    end
end

