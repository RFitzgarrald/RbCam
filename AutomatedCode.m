%% beam movement calculation
clear;clc;
close all
%figure;
set(0,'DefaultFigureWindowStyle','docked')
%Set w and h to the dimensions of the image
w = 1200;
h = 1200;
x = 1:w;
y = 1:h;
res = zeros(1,1);
%Images = cell(1,1);    %creates cell array to store each image 
MaxX = cell(1,1);      %creates cell array for the max X coordinate for each image
MaxY = cell(1,1);      %creates cell array for the max Y coordinate for each image
Y = cell(1,1);         %cell array for Y-value from each fit
X = cell(1,1);         %cell array for X-value from each fit
R2 = cell(1,1);        %cell array for R-squared value from each fit
%Load background data here to have it subtracted from the image
data = matfile('/Volumes/FLASHDRIVE2/WorkData/20200124/Background/background.mat');
backg = data.Z1_filt;
i = 8;                  %number of images to be summed together
%This loop sums 'i' images together, applies the fit, and stores the info;
%it does this until it loops through all of the images
for j = 1:400/i         %1:(total number of images/i)
    Z1_filt = zeros(h,w);
    u = (1-i)+i*j;      
    v = u+i-1;          %set so that k will sum the next successive images and work its way through all of the images
for k = u:v
    %Adjust method of reading images based on file type
    Z1 = fitsread(sprintf('SummedIm_%d.fit',k));
    Z1_filter = Z1-backg;
    Z1_filt = Z1_filt+Z1_filter;
    %Images{k} = Z1_filt;   
end
Z1_summed = Z1_filt;
%Code to find the average of 10 summed background images
%Z1_filt = Z1_filt./10;                             %Average out background 
%save('background.mat','Z1_filt');
%These can be used to apply filters to the image if needed
%Z1_filter=wiener2(Z1_filt,[10 10]);    
%Z1_filter2=imgaussfilt(Z1_filt,1.5);
%Z1_filter = Z1_filt;

%These find the max intensity value which it uses as a starting point for
%the fit
    [max_num, max_idx] = max(Z1_summed(:));
    [Xc,Yc] = ind2sub(size(Z1_summed),max_idx);
    MaxX{j} = Xc;
     MaxY{j} = Yc;
%{     
%% Multiply Raw Data by Another Function
[X2,Y2] = meshgrid(x,y);
c = 100; %The value the function flattens out at will be half this value
l = 0.5; %Height of the function
z = 60; %Controls the flatness of the peak
m = Yc; %Position in X-direction
n = Xc; %Position in Y-direction
t = 20; %width in X-direction
d = 20; %width in Y-direction
Filter = -c*(l*exp(-z*(exp(-((X2-m).^2/(t^2))-((Y2-n).^2/(d^2)))))-1/2);
%}
%Z1_filter_Function = Z1_filter;     %.*Filter;


%{
 %% Manual Gaussian Filter
% Source is:
% https://www.imageeprocessing.com/2014/04/gaussian-filter-without-using-matlab.html 
%Standard Deviation
% sigma = 1.5;
% %Window size
% filter = 2*ceil(2*sigma)+1;
% sz = (filter-1)/2;     %kernel size, using the default that the imgaussfilt uses
% [x2,y2]=meshgrid(-sz:sz,-sz:sz);
% 
% M = size(x2,1)-1;
% N = size(y2,1)-1;
% 
% Exp_comp = -(x2.^2+y2.^2)/(2*sigma*sigma); 
% Kernel= exp(Exp_comp)/(2*pi*sigma*sigma); 
% %Initialize
% I = Z1_filter;
% Output=zeros(size(I));
% %Pad the vector with zeros
% I = padarray(Z1_filter,[sz sz], 'replicate');
% 
% gaussArray = ones(size(I,1),size(I,2));
% gaussArray(Xc-20:Xc+20,Yc-20:Yc+20) = 1.5;  %This currently has a square profile. It could be changed to Gaussian
% %Convolution
% for l = 1:size(I,1)-M
%     for w =1:size(I,2)-N
%         Temp = I(l:l+M,w:w+M).*Kernel;
%         Temp = Temp.*gaussArray(l:l+M,w:w+M);
%         Output(l,w)=sum(Temp(:));
%     end
% end
% %Image without Noise after Gaussian blur
% figure, surf(x,y,Output)
% shading interp
%}

%%

%     Z_fit = Z1_filter(Xc-75:Xc+75, Yc-75:Yc+75); %trims matrix to center on the max values
%     X_fit=Xc-75:Xc+75;
%     Y_fit=Yc-75:Yc+75;
%Trimming the matrix can cause the center position to differ too much
%between consecutive images, so this avoids cutting the matrix
 Z_fit = Z1_summed;
 X_fit = 1:1:w;
 Y_fit = 1:1:h;
    Z_fit=double(Z_fit);
    format long

    [fitresult, gof] = R_createFit(X_fit, Y_fit, Z_fit,Xc,Yc);  %uses fit code to create a fit
    title(sprintf('%d',j));
    coeff = coeffvalues(fitresult);
    Y{j} = coeff(5);                %stores x and y coordinates from the fit's center as well as each R-squared value
    X{j} = coeff(6);
    R2{j} = gof.rsquare;
    norm = max(max(Z1_summed));
    res(j)=mean(abs(Z1_summed-fitresult(x,y)),'all')/(norm);
end
%{
%Code for old Watec camera
for k = 1:1
    image = readtable(sprintf('multiplebeams_setup3_im%d-full.txt',k)); %loads in each image within the loop
    image.Properties.VariableNames = {'NA','NA2','intensity'};      %renames the table variables
    intensity = image.intensity;
    Z = zeros(480,640);
    for j=1:480                         %turns each table into a 480x640 matrix
        for i=1:640
            gogo = (j-1)*640+i;
            Z(j,i) = intensity(gogo);
        end
    end
    Images{k} = Z;                      %stores matrix in cell array
%       Z1=wiener2(Z,[25 25]);     %noise removal
%       Z1=imgaussfilt(Z1,5);      %smoothing
     Z1 = Z;
    [max_num,max_idx] = max(Z1(:));      %finds max X and Y coordinates
    [Xc,Yc]=ind2sub(size(Z1),max_idx);
    MaxX{k} = Xc;                       %stores X and Y coordinates in cell arrays
    MaxY{k} = Yc;
%     Z_fit = Z1(Xc-80:Xc+80,Yc-80:Yc+80); %trims matrix to center on the max values
%     X_fit=Xc-80:Xc+80;
%     Y_fit=Yc-80:Yc+80;
Z_fit = Z1;
X_fit = x;
Y_fit = y;
    format long
    set(0,'DefaultFigureWindowStyle','docked')
    [fitresult, gof] = R_createFit(X_fit, Y_fit, Z_fit);  %uses fit code to create a fit
    title(sprintf('%d',k));
    coeff = coeffvalues(fitresult);
    Y{k} = coeff(5);                %stores x and y coordinates from the fit's center as well as each R-squared value
    X{k} = coeff(6);
    R2{k} = gof.rsquare;
%     norm = max(max(Z));
%     res(k)=mean(abs(Z(X_fit,Y_fit)-Z_fit),'all')/(norm);
end
%}
TotalCell = [MaxY' MaxX' X' Y' R2'];    %concatenates all the cells into one large cell array
Results = cell2table(TotalCell);        %converts cell array to table
Results.Properties.VariableNames = {'Max_Y','Max_X','Fit_X','Fit_Y','R_Squared'};   %renames table variables
hold off
%% Figures
% figure('Name','Data Filter');
% surf(x,y,Filter);
% shading interp
% view([0,90])
% axis equal
% title('Filter Function');
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);

% 
% figure('Name','Data Before Filter');
% surf(x,y,Z1_filter);
% shading interp
% view([0,90])
% axis equal
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);
% title('Before Filter');
% 
% figure('Name','Data After Filter');
% surf(x,y,Z1_filter_Function);
% shading interp
% view([0,90])
% axis equal
% title('After Filter');
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);

% figure('Name','Background');
% surf(x,y,backg);
% shading interp
% view([0,90])
% axis equal
% title('Averaged Background');
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);
% 
% figure('Name','Data W/Background');
% surf(x,y,Z1_filt);
% shading interp
% view([0,90])
% axis equal
% title('Data With Background');
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);

figure('Name','Data Before Fit');
surf(x,y,Z1_summed);
shading interp
view([0,90])
axis equal
title('Data Before Fit');
xlim([x(1),x(end)]);
ylim([y(1),y(end)]);
%%
figure('Name','Centers')
% hold on;
X = cell2mat(X);
Y = cell2mat(Y);
plot(X(1,:),Y(1,:),'o') %plot the center of gaussian fits
% hold off;
title('coordinates of the centers')
%%
% figure ('Name','Original Function');
% surf(x,y,Z1_filt)
% shading interp
% view([0,90])
% axis equal
% title('original function')
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);
%%
% figure ('Name','Smoothed');
% surf(x,y,Z1_filter)
% shading interp
% view([0,90])
% axis equal
% title('smoothed function')
% xlim([x(1),x(end)]);
% ylim([y(1),y(end)]);

%%
figure ('Name','Fit');
plot(fitresult);
%%
% figure ('Name','Final Image');
% % subplot(2,2,2)
% surf(X_fit,Y_fit,abs(Z_fit));%use this as source
% shading interp
% view([0,90])
% axis equal
% title('accumulated and processed image')
% xlim([X_fit(1),X_fit(end)]);
% ylim([Y_fit(1),Y_fit(end)]);
%%
% stdx = std(X);
% stdy = std(Y);
%Audible signal for the end of the code
  load train
  sound(y,Fs)
RSquare = table2array(Results(:,5));
%Check to see if there are fits that are unusually poor
minR = min(RSquare);
X_range = max(X)- min(X);
Y_range = max(Y)- min(Y);
%} 
