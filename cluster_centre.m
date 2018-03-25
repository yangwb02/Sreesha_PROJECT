%function [lb,center] = adaptcluster_kmeans(im)
% This code is written to implement kmeans clustering for segmenting any
% Gray or Color image. There is no requirement to mention the number of cluster for
% clustering. 
% IM - is input image to be clustered.
% LB - is labeled image (Clustered Image).
% CENTER - is array of cluster centers.
% Execution of this code is very fast.
% It generates consistent output for same image.
% Written by Ankit Dixit.
% January-2014.
clc;
clear all;
warning off;
%function [lb,center] = GrayClustering(gray)
a=imread('A.tif');
a=rgb2gray(a);
gray = double(a);
array = gray(:); % Copy value into an array.
% distth = 25;
i = 0;j=0; % Intialize iteration Counters.
tic
while(true)
    seed = mean(array); % Initialize seed Point.
    i = i+1; %Increment Counter for each iteration.
    while(true)
        j = j+1; % Initialize Counter for each iteration.
        dist = (sqrt((array-seed).^2)); % Find distance between Seed and Gray Value.
        distth = (sqrt(sum((array-seed).^2)/numel(array)));% Find bandwidth for Cluster Center.
        %         distth = max(dist(:))/5;
        qualified = dist<distth;% Check values are in selected Bandwidth or not.
        newseed = mean(array(qualified));% Update mean.
                if isnan(newseed) % Check mean is not a NaN value.
            break;
        end
         if seed == newseed || j>10 % Condition for convergence and maximum iteration.
            j=0;
            array(qualified) = [];% Remove values which have assigned to a cluster.
            center(i) = newseed; % Store center of cluster.
            break;
        end
        seed = newseed;% Update seed.
     end
    
    if isempty(array) || i>10 % Check maximum number of clusters.
        i = 0; % Reset Counter.
        break;
    end
    
end
toc

center = sort(center); % Sort Centers.
newcenter = diff(center);% Find out Difference between two consecutive Centers. 
intercluster = (max(gray(:)/10));% Findout Minimum distance between two cluster Centers.
center(newcenter<=intercluster)=[];% Discard Cluster centers less than distance.

% Make a clustered image using these centers.

vector = repmat(gray(:),[1,numel(center)]); % Replicate vector for parallel operation.
centers = repmat(center,[numel(gray),1]);

distance = ((vector-centers).^2);% Find distance between center and pixel value.
[~,lb] = min(distance,[],2);% Choose cluster index of minimum distance.
lb = reshape(lb,size(gray));% Reshape the labelled index vector.
figure
imshow(lb,[]);
disp(center);

