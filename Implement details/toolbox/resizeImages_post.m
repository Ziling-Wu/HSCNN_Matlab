function imds = resizeImages_post(imds, imageFolder)
% Resize images to [360 480].

if ~exist(imageFolder,'dir') 
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder);
    return; % Skip if images already resized
end

reset(imds)
index =1;
% while hasdata(imds)
while index<10
    index = index+1;
    % Read an image.
    [I,info] = read(imds);
    I = double(I);
    I = repmat(I,[1,1,3]);
    Ibar = mean(I(:));
    Istd = std(I(:));
    I = (I-Ibar)./Istd;
    I = -I;
%     % Resize image.
% %     I = imresize(I,[360 480]);    
%     % Write to disk.
%     I = (I-min(I(:)))./(max(I(:))-min(I(:)));
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder filename ext])
end

imds = imageDatastore(imageFolder);
end