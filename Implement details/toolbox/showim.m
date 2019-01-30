function showim(a,b,cs1)

global xx

imagesc(xx,xx,a)
axis square 

    if nargin < 2 
        colormap parula
    elseif b==1
        colormap gray
           else
        colormap parula
    end
    
    if nargin>2
        caxis(cs1);
    end 
    %colorbar
%     xlabel('x(mm)')
%     ylabel('y(mm)')
set(gca,'fontsize', 20);
axis tight
colormap gray
% colorbar
set(gcf,'color','white')
axis off
end