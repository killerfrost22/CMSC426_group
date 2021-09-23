clc
clear all
close all
%% hyperparermeters
thresh = .000008;
k = 6;
%% training
% read images
path = dir('./train_images/*.jpg');
filenames={path.name};
%filenames=filenames(1:2);
num_train_images = length(filenames);
orange = [];
curr = zeros(1,3);
for i = 1:num_train_images
    name_full = filenames{1,i};
    name = name_full(1:length(name_full)-4);
    img = imread(strcat('./train_images/', name,'.jpg'));
    mask = imread(strcat('./train_images/masks/', name,'-mask.bmp'));
    [rows, cols, colorChannels] = size(img);
    
    for i=1:rows
    for j=1:cols
        if mask(i,j) == 255
            curr(1,1) = img(i,j,1);
            curr(1,2) = img(i,j,2);
            curr(1,3) = img(i,j,3);
            orange = vertcat(orange,double(curr));
        end
    end
    end
    
end
x = orange;
[mu, sigma] = trainGMM(x, k);
plotGMM(mu, sigma, 1, x);
%% testing
path = dir('./test_images/*.jpg');
test_filenames={path.name};
num_of_test_img = length(test_filenames);
for i = 1:num_of_test_img
    currImg = imread(strcat('./test_images/', test_filenames{i}));
    x = currImg;    
    cluster = testGMM(x, mu, sigma, thresh, k);    
    % Measure Depth
    d = measureDepth(cluster');
    figure();
    imshow(cluster'); 
    disp('distance of the ball:' + d);    
end
%% plot figure related function 
function a = plotGMM(mu, sigma, overlay, data)
    figure;
    if (overlay > 0)
        plot3(data(:,1), data(:,2), data(:,3), 'bo')
    end
    h1 = plot_gaussian_ellipsoid(mu(1,:), sigma(:,:,1));
    plot_gaussian_ellipsoid(mu(2,:), sigma(:,:,2));
    plot_gaussian_ellipsoid(mu(3,:), sigma(:,:,3));
    plot_gaussian_ellipsoid(mu(3,:), sigma(:,:,4));
    plot_gaussian_ellipsoid(mu(3,:), sigma(:,:,5));
    plot_gaussian_ellipsoid(mu(3,:), sigma(:,:,6));
    plot_gaussian_ellipsoid(mu(3,:), sigma(:,:,7));
    set(h1,'facealpha',0.6);
    %view(129,36); set(gca,'proj','perspective'); 
    %grid on; 
    grid on; axis equal; axis tight;

end
function h = show3d(means, C, sdwidth, npts, axh)
    if isempty(npts), npts=20; end
    [x,y,z] = sphere(npts);
    ap = [x(:) y(:) z(:)]';
    [v,d]=eig(C); 
    if any(d(:) < 0)
        d = max(d,0);
    end
    d = sdwidth * sqrt(d); 
    bp = (v*d*ap) + repmat(means, 1, size(ap,2)); 
    xp = reshape(bp(1,:), size(x));
    yp = reshape(bp(2,:), size(y));
    zp = reshape(bp(3,:), size(z));
    h = surf(axh, xp,yp,zp)
end

function h = plot_gaussian_ellipsoid(m, C, sdwidth, npts, axh)
    if ~exist('sdwidth', 'var'), sdwidth = 1; end
    if ~exist('npts', 'var'), npts = []; end
    if ~exist('axh', 'var'), axh = gca; end
    if numel(m) ~= length(m), 
        error('M must be a vector'); 
    end
    if ~( all(numel(m) == size(C)) )
        error('Dimensionality of M and C must match');
    end
    if ~(isscalar(axh) && ishandle(axh) && strcmp(get(axh,'type'), 'axes'))
        error('Invalid axes handle');
    end
    set(axh, 'nextplot', 'add');
    switch numel(m)
    case 2, h=show2d(m(:),C,sdwidth,npts,axh);
    case 3, h=show3d(m(:),C,sdwidth,npts,axh);
    otherwise
    error('Unsupported dimensionality');
    end
    if nargout==0,
        clear h;
    end
end
