function imds = resizeImages_concatenate(imds, imageFolder)
% Resize images to [360 480].

if ~exist(imageFolder,'dir') 
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder);
    return; % Skip if images already resized
end

reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);
    I = double(I);
    Inew = zeros(size(I,1),size(I,2),3);
    Ibar = mean(I(:));
    Istd = std(I(:));
    Inor = (I-Ibar)./Istd;
    Ipos = Inor.*(Inor>0);
    Ineg = -Inor.*(Inor<0);
%     % Resize image.
% %     I = imresize(I,[360 480]);    
%     % Write to disk.
    Inew(:,:,1) = (I-min(I(:)))./(max(I(:))-min(I(:)));%original
    Inew(:,:,2) = (Ipos-min(Ipos(:)))./(max(Ipos(:))-min(Ipos(:)));%original
    Inew(:,:,3) = (Ineg-min(Ineg(:)))./(max(Ineg(:))-min(Ineg(:)));%original
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(Inew,[imageFolder filename ext])
end

imds = imageDatastore(imageFolder);
end