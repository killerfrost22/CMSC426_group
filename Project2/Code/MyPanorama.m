function [pano] = MyPanorama()

%% YOUR CODE HERE.
% Must load images from ../Images/Input/
% Must return the finished panorama.
sprintf('Hi there!')

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

