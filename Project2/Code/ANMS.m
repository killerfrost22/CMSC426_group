location = "..\Images\Set1\";
curr_img = imread(location + "1.jpg");
next_img = imread(location + "2.jpg");

all=dir(location + '*.jpg');
num_images = size(all,1);

imageSize = empty(num_images,2)
for I = 2: all
    curr_img_grayscale = rgb2gray(curr_img);
    next_img_grayscale = rgb2gray(next_img);
    
    imageSize(I,:) = size(curr_img_grayscale)
end

function p = ANMS(I, NBest, SHOW_OUTPUT) 
    cornerScoreImg = cornermetric(I); 
    features = imregionalmax(cornerScoreImg);
    sz = size(I); 
    x = [];
    y = [];
    for i = 20:sz(1)-20
        for j = 20:sz(2)-20
            if(features(i,j) == 1)
                x = [x; j];
                y = [y; i];
            end
        end
    end   
    for i = 1:NStrong
     for j = 1:NStrong
        % We check if the metric scores are bigger between the current
        % point and the previous, if it is we then get the distances. We
        % then iteratively get smaller distances.
        if cornerScoreImg(y(j), x(j)) > cornerScoreImg(y(i), x(i))
          % Calculate distance
          ED = (y(j) - y(i))^2 + (x(j) - x(i))^2;
          if (ED < radius(i))
            radius(i) = ED;
          end
        end
     end
    end
    %sort the radius descending order
    [~,idx] = sort(radius, 'descend');
    %The amount will be assigned the 
    if SHOW_OUTPUT
        imshow(I)
        hold on
        plot(features)
        hold off
    end
end

 function [] = displayANMS(img, x_best, y_best)
     imshow(img);
     hold on
     
     plot(x_best, y_best, 'r.');
     hold off
 end