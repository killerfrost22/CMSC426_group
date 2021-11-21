function [NewLocalWindows] = localFlowWarp(WarpedPrevFrame, CurrentFrame, LocalWindows, Mask, WindowWidth)
% LOCALFLOWWARP Calculate local window movement based on optical flow between frames.
% rto do 
flow_multiplier = 1.5;
numLocalWindows = size(LocalWindows, 1);
% Calculate optical flow for entire image
opticFlow = opticalFlowFarneback;
gray1 = rgb2gray(WarpedPrevFrame);
gray2 = rgb2gray(CurrentFrame);
estimateFlow(opticFlow,gray1);
flow = estimateFlow(opticFlow,gray2);

NewLocalWindows = zeros(size(LocalWindows));
% Update each local window
for i=1:numLocalWindows
    x = round(LocalWindows(i,1));
    y = round(LocalWindows(i,2));
    if x-WindowWidth/2 < 1 || x+(WindowWidth/2-1) > size(gray2,2) || y-WindowWidth/2 < 1 || y+(WindowWidth/2-1) > size(gray2,1)
        continue
    end
    yRange = (y-(WindowWidth/2)):(y+(WindowWidth/2 - 1));
    xRange = (x-(WindowWidth/2)):(x+(WindowWidth/2 - 1));
    N_prime_mask = false(size(gray2));
    N_prime_mask(yRange,xRange) = true;
    local_interior_mask = Mask & N_prime_mask;
    local_interior_flow = [flow_multiplier*flow.Vx(local_interior_mask), flow_multiplier*flow.Vy(local_interior_mask)];
    average_local_flow = mean(local_interior_flow);
    NewLocalWindows(i,:) = round(LocalWindows(i,:) + average_local_flow);
    if size(local_interior_flow, 1) == 0
        NewLocalWindows(i,:) = round(LocalWindows(i,:));
    end
end
%     
%     figure;
%     imshow(current_frame);
%     hold on
%     showLocalWindows(NewLocalWindows, Width,'r.');
%     showLocalWindows(LocalWindows, Width,'b.');
%     hold off
end

