%% Operate
img_s = imread('col.png');
sz = size(img_s);
mask = imread('mas.png');
mask = rgb2gray(mask);
%mask = zeros(sz(1),sz(2));
%{
for i=15:35
    for j=15:35
        mask(i,j,1)= 255;
    end
end    
%}
img_p = padarray(img_s,[4 4],'replicate','both');
%% vis
figure(1);
imshow(img_s);
figure(2);
imshow(mask);
%imshow(img_p);
disp(size(img_s));
%% Grad
[ix1,iy1] = gradient(double(img_s(:,:,1)));
[ix2,iy2] = gradient(double(img_s(:,:,2)));
[ix3,iy3] = gradient(double(img_s(:,:,3)));
%disp(size(ix));
%ix = abs(ix1)+abs(ix2)+abs(ix3);
%iy = abs(iy1)+abs(iy2)+abs(iy3);
ix = ix1+ix2+ix3;
iy = iy1+iy2+iy3;
figure(3);
imshow(ix);
figure(4);
imshow(iy);

%% Rotation
temp = ix;
ix = -iy;
iy = temp;
ix = (ix)/(255);
iy = (iy)/(255);
%ix = im2bw(ix,0.1);
%iy = im2bw(iy,0.1);
%disp(ix/255);
figure(5);
imshow(ix);
figure(6);
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
patch_size = 9.0;
sz = size(img_s);
D = repmat(0.001,size(mask));
img_p = padarray(img_s,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
img_c = padarray(img_s,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
C_new = padarray(C,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');    % 233x233 confidence
C_old =  padarray(C,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
C_nc = C_new;
img_pnc = img_p;
while(any(mask(:)))
    %{
    [ix1,iy1] = imgradientxy(double(img_s(:,:,1)));
    [ix2,iy2] = imgradientxy(double(img_s(:,:,2)));
    [ix3,iy3] = imgradientxy(double(img_s(:,:,3)));
    %disp(size(ix));
    ix = abs(ix1)+abs(ix2)+abs(ix3);
    iy = abs(iy1)+abs(iy2)+abs(iy3);
    temp = (ix);
    ix = (iy);
    iy = temp;
    ix = (ix)/(255*3);
    iy = (iy)/(255*3);
    %}
    %delta =  conv2(mask,[1,1,1;1,-8,1;1,1,1]);
    delta =  edge(mask,'approxcanny');
    %figure();
    %imshow(delta);
    [cnx, cny] = find(abs(delta)>0);
    %disp(cnx);
    [Npx1,Npy1] = gradient(double(mask));
    [Npx2,Npy2] = gradient(255-double(mask));
    %figure();
    %imshow(Npx1);
    %disp(Npx1);
    Npx =abs(double((Npx1)-(Npx2))/(255*255));
    Npy =abs(double((Npy1)-(Npy2))/(255*255));
    Npx = im2bw(Npx,0.000001);
    Npy = im2bw(Npy,0.000001);
    figure(7);
    imshow(abs(Npx));
    figure(8);
    imshow(abs(Npy));
    %Npx =double(Npx2/255);
    %Npy =double(Npy2/255);
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
        D(cn(i,1),cn(i,2)) = D(cn(i,1),cn(i,2))+10*abs(ix(cn(i,1),cn(i,2)).*N_delta(i,1))+10*abs((iy(cn(i,1),cn(i,2)).*N_delta(i,2)));

        %disp(N_delta(i));
    end
    figure(5);
    imshow(abs(D));
    %disp(size(N_delta));

    %D(cn) = abs(ix(cn).*N_delta(:,1))+abs(iy(cn).*N_delta(:,2));

    % confidence
    patch_size = 9.0;
    sum_p=0.0;
    max_ind = floor(patch_size/2);
    
    C_new = padarray(C,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');    % 233x233 confidence
    %C_old =  padarray(C,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');   % 54x54
    
    for i=1:length(cn)
        midx = cn(i,1);
        midy = cn(i,2);
        for j=(midx-floor(patch_size/2)):(midx+floor(patch_size/2))
            for k=(midy-floor(patch_size/2)):(midy+floor(patch_size/2))
                sum_p = sum_p + C_new(j,k);
            end
        end
        C_new(midx+max_ind,midy+max_ind) = (double(sum_p)/81.0);
        sum_p=0.0;
    end
    %C_new = padarray(C,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
    
    %figure();
    %imshow(C);
    % Priority
    C_ch = C_new(max_ind+1:sz(1)+max_ind,max_ind+1:sz(2)+max_ind);          % 50x50
    P = C_ch.*D;
    cur_patch = [cn(1,1),cn(1,2)];        % 50-50
    for i=1:length(cn)
        if (abs(P(cur_patch(1),cur_patch(2)))<abs(P(cn(i,1),cn(i,2))))
            cur_patch(1) = cn(i,1);
            cur_patch(2) = cn(i,2);
        end
    end

    % Best exemplar
    
    
    temp_img = zeros(patch_size,patch_size);
    overlap = 100000;
    patch_img = img_s(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind,:);
    C_patch = C_old(cur_patch(1)+max_ind:cur_patch(1)+3*max_ind,cur_patch(2)+max_ind:cur_patch(2)+3*max_ind);
    C_patch = im2bw(C_patch,0.0001);
    %figure(9);
    %imshow(C_patch);  
    for i=max_ind+1:sz(1)+max_ind
        for j=max_ind+1:sz(2)+max_ind
            temp_img = img_pnc(i-max_ind:i+max_ind,j-max_ind:j+max_ind,:);
            count = 0.0;
            C_img = C_nc(i-max_ind:i+max_ind,j-max_ind:j+max_ind);
            %C_img = im2bw(C_img,0.1);
            tar_pat = 1;
            for t=1:patch_size
                for s=1:patch_size
                    if(C_img(t,s)==0)
                       tar_pat = 0;
                    end
                end
            end
            
            %disp(i);
            if(tar_pat==1)  

                for m=1:patch_size
                    for n=1:patch_size
                        if(C_patch(m,n)>0.0)
                            a1 = temp_img(m,n,1);
                            a2 = temp_img(m,n,2);
                            a3 = temp_img(m,n,3);
                            b1 = patch_img(m,n,1);
                            b2 = patch_img(m,n,2);
                            b3 = patch_img(m,n,3);
                            %count = count+sum(square(temp_img(m,n,:)-patch_img(m,n,:)));
                            temp_ls = [a1,a2,a3];
                            patch_ls = [b1,b2,b3];
                            
                            count = count+sum(abs(temp_ls-patch_ls));
                        
                        end
                    end
                end
                if(count<overlap)
                    %disp(count);
                    %disp(temp_img);
                    %disp(patch_img);
                    overlap = count;
                    ptx = i;
                    pty = j;
                    %disp(ptx);
                    %disp(pty);
                end
            end
            %disp(overlap);
        end
    end

    % Replace best patch
%%    
    img_c(cur_patch(1):cur_patch(1)+2*max_ind,cur_patch(2):cur_patch(2)+2*max_ind,1) = img_pnc(ptx-max_ind:ptx+max_ind,pty-max_ind:pty+max_ind,1);
    img_c(cur_patch(1):cur_patch(1)+2*max_ind,cur_patch(2):cur_patch(2)+2*max_ind,2) = img_pnc(ptx-max_ind:ptx+max_ind,pty-max_ind:pty+max_ind,2);
    img_c(cur_patch(1):cur_patch(1)+2*max_ind,cur_patch(2):cur_patch(2)+2*max_ind,3) = img_pnc(ptx-max_ind:ptx+max_ind,pty-max_ind:pty+max_ind,3);
    figure(12);
    imshow(img_pnc(ptx-max_ind:ptx+max_ind,pty-max_ind:pty+max_ind,:))
    img_im = img_c(max_ind+1:sz(1)+max_ind,max_ind+1:sz(2)+max_ind,:);
    figure(9);
    imshow(img_im);
%%
    % Update confidence and mask

    mask(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind) = 0;
    figure(10);
    imshow(mask);
    C_old(cur_patch(1):cur_patch(1)+2*max_ind,cur_patch(2):cur_patch(2)+2*max_ind) = C_new(ptx:ptx+2*max_ind,pty:pty+2*max_ind);
    %C = im2bw(C,0.0001);
    ix_ch = padarray(ix,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
    iy_ch = padarray(iy,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
    ix(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind) = ix_ch(ptx-max_ind:ptx+max_ind,pty-max_ind:pty+max_ind);
    iy(cur_patch(1)-max_ind:cur_patch(1)+max_ind,cur_patch(2)-max_ind:cur_patch(2)+max_ind) = iy_ch(ptx-max_ind:ptx+max_ind,pty-max_ind:pty+max_ind);
    %imshow(mask);
    C = C_old(max_ind+1:sz(1)+max_ind,max_ind+1:sz(2)+max_ind);
    %C = im2bw(C,0.0001);
    %C_old = padarray(C,[floor(patch_size/2) floor(patch_size/2)],'replicate','both');
end

img_last = img_im;
imshow(img_last);

%%
figure(9);
imshow(img_im);
%imwrite(img_im,'bird_inp3.jpg','jpg');