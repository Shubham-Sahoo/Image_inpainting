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
source = ~mask;
C = double(source);
D = repmat(-0.0001,size(mask));

%while(any(mask(:)))
%delta =  conv2(mask,[1,1,1;1,-8,1;1,1,1]);
delta =  edge(mask,'approxcanny');
figure();
imshow(delta);
[cnx, cny] = find(delta>0);
%disp(cnx);
[Npx1,Npy1] = imgradientxy(mask);
[Npx2,Npy2] = imgradientxy(~mask);
%figure();
%imshow(Npx1);
%disp(Npx1);
Npx =double((Npx1-Npx2)/(255));
Npy =double((Npy1-Npy2)/(255));
disp(size(Npx));
figure();
imshow(abs(255*Npx));
figure();
imshow(abs(255*Npy));
cn = [cnx cny];
N_delta = zeros(size(cn));
for i=1:length(cn)
    N_delta(i,1) = Npx(cn(i,1),cn(i,2)); 
    N_delta(i,2) = Npy(cn(i,1),cn(i,2));
    D(cn(i,1),cn(i,2)) = 100*abs(ix(cn(i,1),cn(i,2))*N_delta(i,1))+abs(iy(cn(i,1),cn(i,2))*N_delta(i,2));
    %disp(N_delta(i));
end
disp(size(N_delta));

%D(cn) = abs(ix(cn).*N_delta(:,1))+abs(iy(cn).*N_delta(:,2));

%% patch
for c = cn'
   p = patch();
end

%end


