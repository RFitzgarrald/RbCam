%% beam movement calculation
%Load background data here to have it subtracted from the image
tic
clear;clc;close all
data = matfile('/Volumes/FLASHDRIVE2/WorkData/20200124/Background/background.mat');
backg = data.Z1_filt;
set(0,'DefaultFigureWindowStyle','docked')
%Set w and h to the dimensions of the image
w = 1200;
h = 1200;
x = 1:w;
y = 1:h;
X_fit = 1:1:w;
Y_fit = 1:1:h;
res = zeros(400,1);
Images = cell(400,1);
Sums = cell(400,1);
Rsquare = zeros(400,1);
X_coord = zeros(400,1);
Y_coord = zeros(400,1);
%% Load Images into Array
for s = 1:400
    Im = fitsread(sprintf('SummedIm_%d.fit',s));
    Images{s} = Im;
end
%% Sum Images and Store
totalSum = zeros(h,w);
for t = 1:400
    noBG = Images{t} - backg;
    totalSum = totalSum + noBG;
    Sums{t} = totalSum;
end
%% Process Images
%l determines the number of image combinations you want to check
%{
for l = 215:400
Z1_summed = Sums{l};
%These find the max intensity value which it uses as a starting point for
%the fit
    [max_num, max_idx] = max(Z1_summed(:));
    [Xc,Yc] = ind2sub(size(Z1_summed),max_idx);
    Z_fit = Z1_summed;
    Z_fit=double(Z_fit);
    format long   
    [fitresult, gof] = R_createFit(X_fit, Y_fit, Z_fit,Xc,Yc);  %uses fit code to create a fit
    title(sprintf('%d,%d',l,j));
    coeff = coeffvalues(fitresult);
    norm = max(max(Z1_summed));
    res(l)=mean(abs(Z1_summed-fitresult(x,y)),'all')/(norm);
end
%}
for l = 1:400
    Data = Images{l};
    [max_num, max_idx] = max(Data(:));
    [Xc,Yc] = ind2sub(size(Data),max_idx);
    Z_fit = double(Data);
    format long
    [fitresult, gof] = R_createFit(X_fit, Y_fit, Z_fit,Xc,Yc);  %uses fit code to create a fit
    title(sprintf('%d',l));
    coeff = coeffvalues(fitresult);
    X_coord(l) = coeff(5);
    Y_coord(l) = coeff(6);
    Rsquare(l) = gof.rsquare;
    norm = max(max(Data));
    res(l)=mean(abs(Data-fitresult(x,y)),'all')/(norm);
end
hold off
%% Figures
toc