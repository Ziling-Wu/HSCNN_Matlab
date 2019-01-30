function im = segmentation(im)
    
% im = im*10^3;
%% binary
GMM = fitgmdist(im(:),2);
divide_percentage = max(GMM.ComponentProportion);
divide_loc = quantile(im(:),divide_percentage);
if divide_loc<0
    temp = GMM.mu;
    divide_loc = temp(temp>0);
end
% divide_loc=0.00327;
im(im<divide_loc)=0;
im(im>=divide_loc)=1;
% %%
% f = im;
% %% median filter
% f1 = medfilt2(f,[6,6]);
% %imagesc(f1)
% %% get rid of small artifacts
% f2=bwareaopen(f1,60);
% %imagesc(f2)
% %%
% im1 = ~bwareaopen(~f2,50);
