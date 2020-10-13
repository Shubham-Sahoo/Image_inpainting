%% Operate
img_s = imread('download.jpg');

%mask = imread('square.jpg');
mask = zeros(225);
for i=100:150
    for j=100:150
        mask(i,j)= 1;
    end
end    
img_p = padarray(img_s,[4 4],'replicate','both');
%% vis
figure();
imshow(img_s);
figure();
imshow(mask);
%imshow(img_p);
disp(size(img_s));
%% Grad
[ix,iy] = gradient(double(img_s(:,:,1)));
%disp(size(ix));
figure();
imshow(ix);
figure();
imshow(iy);

%% Rotation
temp = ix;
ix = -iy;
iy = temp;
ix = (ix+1)/510;
iy = (iy+1)/510;
%disp(ix/255);
figure();
imshow(ix);
figure();
imshow(iy);
%% Gaussian
%ig = imgaussfilt(double(img_s(:,:,1)),2);
%disp(ig);
%figure();
%imshow(ig/255);
%% Normal
img_im = img_s;
source = ~mask;                         % 225x225 source region
C = double(source);
D = repmat(-0.0001,size(mask));
C_new = padarray(C,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');    % 233x233 confidence

while(any(mask(:)))
    %delta =  conv2(mask,[1,1,1;1,-8,1;1,1,1]);
    delta =  edge(mask,'approxcanny');
    %figure();
    %imshow(delta);
    [cnx, cny] = find(delta>0);
    %disp(cnx);
    [Npx1,Npy1] = imgradientxy(mask);
    [Npx2,Npy2] = imgradientxy(~mask);
    %figure();
    %imshow(Npx1);
    %disp(Npx1);
    Npx =double((Npx1-Npx2)/(255));
    Npy =double((Npy1-Npy2)/(255));
    %disp(size(Npx));
    %figure();
    %imshow(abs(255*Npx));
    %figure();
    %imshow(abs(255*Npy));
    cn = [cnx cny];
    N_delta = zeros(size(cn));
    for i=1:length(cn)
        N_delta(i,1) = Npx(cn(i,1),cn(i,2)); 
        N_delta(i,2) = Npy(cn(i,1),cn(i,2));
        D(cn(i,1),cn(i,2)) = 100*abs(ix(cn(i,1),cn(i,2))*N_delta(i,1))+abs(iy(cn(i,1),cn(i,2))*N_delta(i,2));

        %disp(N_delta(i));
    end
    %disp(size(N_delta));

    %D(cn) = abs(ix(cn).*N_delta(:,1))+abs(iy(cn).*N_delta(:,2));

    % confidence
    patch_size = 9.0;
    sum_p=0.0;
 
    for i=1:length(cn)
        midx = cn(i,1);
        midy = cn(i,2);
        for j=(midx-floor(patch_size/2)):(midx+floor(patch_size/2))
            for k=(midy-floor(patch_size/2)):(midy+floor(patch_size/2))
                sum_p = sum_p + C_new(j,k);
            end
        end
        C(midx,midy) = (double(sum_p)/81.0);
        sum_p=0.0;
    end
    C_new = padarray(C,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
    %figure();
    %imshow(C);
    % Priority
    P = C.*D;
    cur_patch = [cn(1,1),cn(1,2)];
    for i=1:length(cn)
        if (abs(P(cur_patch(1),cur_patch(2)))<abs(P(cn(i,1),cn(i,2))))
            cur_patch(1) = cn(i,1);
            cur_patch(2) = cn(i,2);
        end
    end

    % Best exemplar
    sz = size(img_s);
    img_p = padarray(img_s,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
    temp_img = zeros(patch_size,patch_size);
    overlap = 100000;
    max_ind = floor(patch_size/2);
    patch_img = img_p(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind);
    C_patch = C_new(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind);
    for i=1:sz(1)-max_ind
        for j=1:sz(2)-max_ind
            temp_img = img_p(i:i+2*max_ind,j:j+2*max_ind);
            count = 0.0;
            C_img = C_new(i:i+2*max_ind,j:j+2*max_ind);
            tar_pat = any(C_img,'all');   
            %disp(i);
            if(tar_pat)     
                for m=1:patch_size
                    for n=1:patch_size
                        if(C_patch(m,n)>0)
                            count = count+abs(temp_img(m,n)-patch_img(m,n));
                        end    
                    end
                end
                if(count<overlap)
                    %disp(C_patch);
                    %disp(temp_img);
                    %disp(patch_img);
                    overlap = count;
                    ptx = i;
                    pty = j;
                end
            end

        end
    end

    % Replace best patch
    img_im(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind,1) = img_p(ptx:ptx+2*max_ind,pty:pty+2*max_ind);
    img_im(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind,2) = img_p(ptx:ptx+2*max_ind,pty:pty+2*max_ind);
    img_im(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind,3) = img_p(ptx:ptx+2*max_ind,pty:pty+2*max_ind);
    %imshow(img_im);

    % Update confidence and mask

    mask(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind) = 0;

    C_new(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind) = C_new(ptx:ptx+2*max_ind,pty:pty+2*max_ind);
    ix_ch = padarray(ix,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
    iy_ch = padarray(iy,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
    ix(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind) = ix_ch(ptx:ptx+2*max_ind,pty:pty+2*max_ind);
    iy(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind) = iy_ch(ptx:ptx+2*max_ind,pty:pty+2*max_ind);
    %imshow(mask);

end

img_last = img_im;
imshow(img_last);

%%
imshow(img_im);
