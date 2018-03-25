
%% TESTING PROGRAM

%%--IMAGE PRE-PROCESSING--%%%%%%
clear all; 
close all; 
clc;
load SVMstruct.mat
[filename, pathname] = uigetfile({'*.dcm';'*.bmp';'*.png';'*.*'}, 'Pick an Image File');
I = dicomread([pathname,filename]);
feature=[];
featureVec = [];
I=imresize(I,[256,256]);
% I=double(I);
imshow(I,[]);title('Original Image');
I=histeq(I);
imshow(I,[]);title('Histogram Equalization')
sigma=1;
smask = fspecial('gaussian', ceil(3*sigma), sigma);%%guassian filter
I = filter2(smask, I, 'same');%%2D digital filter like finite impulse response
imshow(I,[]);title('Gaussian Smoothening')
%[I_1,CH,CV,CD] = dwt2(I,'haar');
%imshow(I_1,[]);title('2D Wavelet Decomposition');
%I_1 = double(I);
%alpha=0.4; beta=0.2; gamma=1; kappa=0.15; wl=0.3; we=0.4; wt=0.7; iterations=100;
%[xs, ys] = select_points(I_1);
%smth =active_contour(I, xs, ys, alpha, beta, gamma, kappa, wl, we, wt, iterations);

%%
%%%%%%%%%%%%%%--ADDITIONAL NOISE REMOVAL--%%%%%%%%%%%%
%Img=imresize(bw,[256,256]);
%figure;imshow(Img);
[R C] = size(I);
smth=I;
for i = 2:R-1
    for j = 2:C-1
        clear tmp;
        tmp = smth(i-1:i+1,j-1:j+1);%3 x 3 matrix
        flg = 0;    
       P = sort(tmp(:));
        if smth(i,j) == 0 || smth(i,j) == 255
            flg = 1;
        end
        if flg == 0 % if the Pixel is normal.
            if P(1) < smth(i,j) && smth(i,j) < P(5) && 0 < P(1) && P(9) < 255 % The 
            %P(X,Y) is an normal pixel if  Pmin < P(X,Y) < Pmax,
            %Pmin > 0 and Pmax < 255; the pixel is not changed. O
            %If above condition fails,P(X,Y) is a noisy pixel.
               smth(i,j) = smth(i,j);
            end
        else        % if the Pixel is noisy.
            if P(1) < P(5) && P(5) < P(9) && 0 < P(5) && P(5) < 255 % If P(X,Y)
                % is a noisy pixel, it is replaced by its middle value 
                % if Pmin < Pmiddle < Pmax and 0 < Pmiddle < 255.
                smth(i,j) = P(5);
            end
            if P(1) >= P(5) || P(5) >= P(9) || P(5) == 255 && P(5) == 0
            % If Pmin < Pmiddle < Pmax  is not satisfied or 255 < Pmiddle = 0,
            %then is a noisy pixel. In this case, 
            %the P(X,Y) is replaced by the value of left neighborhood pixel value.
                smth(i,j) = smth(i,j-1);
            end
        end
    end
end
imshow(smth);
pause(1);
I=smth;
%%
%%%%%%%%%%%%%%%%%%%ACTIVE CONTOUR MODEL%%%%%%%%%%%%%%
m = zeros(size(I,1),size(I,2));          %-- create initial mask
m(70:200,70:200) = 1;
%I = imresize(I,.5);  %-- make image smaller 
%m = imresize(m,.5);  %     for fast computation
imshow(m); title('Mask Initialization');
bw = activecontour(smth,m,1000);
imshow(bw); title('Segmentation');
%%
%%%%%-MMTD FCM-%%%%%%%
I = im2uint8(bw(:));
%%Step 0: Loading and Visualizing Image Randomly 
    train_data = I;
    [dataRow, dataCol] = size(train_data);
    somRow = 20;
    somCol = 20;
% Number of iteration for convergence
    Iteration = 90;
%Parameter Setting  
% Initial size of topological neighbourhood of the winning neuron
    width_Initial = 15;
% Time constant for initial topological neighbourhood size
    t_width = Iteration/log(width_Initial);
% Initial time-varying learning rate
    learningRate_Initial = 0.5;
% Time constant for the time-varying learning rate
    t_learningRate = Iteration;
%% Step 1: Initialize The Weight Of Neurons Randomly 
    fprintf('\nInitializing The mmtd ...\n')
    somMap = randInitializeWeights(somRow,somCol,dataCol);
%% Step 2: Training SOM Iteratively  
    for t = 1:Iteration
    % Size of topological neighbourhood of the winning neuron at the
    % iteration of i
    width = width_Initial*exp(-t/t_width);
    width_Variance = width^2;
    % The time-varying learning rate at the iteration of i
    learningRate = learningRate_Initial*exp(-t/t_learningRate);
    % Prevent learning rate become too small
    if learningRate <0.025 %Threshold value is 0.025
         learningRate = 0.1;
     end
  %% The Competitive Process  
    % Compute the Euclidean distance between each neuron and input
     [euclideanDist, index] = findBestMatch( train_data, somMap, somRow, ...
                                            somCol, dataRow, dataCol );
    
    % Find the index of winning neuron
        [minM,ind] = min(euclideanDist(:));
        [win_Row,win_Col] = ind2sub(size(euclideanDist),ind);
     %% Return the index of winning neuron
     %%  The End of Competitive Process  
     %%  The Cooperative Process 
    % Compute the neighborhood function for each neuron
        neighborhood = computeNeighbourhood( somRow, somCol, win_Row, ...
                                            win_Col, width_Variance);
    
    %% Return the lateral distance between each neuron and winning neuron
   %%  The Adaptive Process 
    % Update the weight of all the neuron on the grid
    %current_Weight_Vector = reshape(somMap(r,c,:),1,dataCol);
        somMap = updateWeight( train_data, somMap, somRow, somCol, ...
                            dataCol, index, learningRate, neighborhood);
             
    %%Illustrate The Updated Clustering Results  
    % Weight vector of neuron
         dot = zeros(somRow*somCol, dataCol);
    % Matrix for SOM plot grid
         matrix = zeros(somRow*somCol,1);
    % Matrix for SOM plot grid for deletion
         matrix_old = zeros(somRow*somCol,1);
            ind = 1;  
    %hold on;
    %f1 = figure(1);
    %set(f1,'name',strcat('Iteration #',num2str(t)),'numbertitle','off');
    % Retrieve the weight vector of neuron
               for r = 1:somRow
                       for c = 1:somCol      
                            dot(ind,:)=reshape(somMap(r,c,:),1,dataCol);
                           ind = ind + 1;
                       end
               end
    end
%%   
%%%%%%FINDING CLUSTER HEADS%%%%%%%%%%

    array = dot(:); % Copy value into an array.
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
gray = double(bw);
center = sort(center); % Sort Centers.
newcenter = diff(center);% Find out Difference between two consecutive Centers. 
intercluster = (max(gray(:)/5));% Findout Minimum distance between two cluster Centers.
center(newcenter<=intercluster)=[];% Discard Cluster centers less than distance.
array = gray(:); % Copy value into an array.
Wv=double(min(center));
Wb=double(max(center));
%c1=min(center);
%c2=max(center);
W=cat(2,Wv,Wb);
vector = repmat(gray(:),[1,numel(W)]); % Replicate vector for parallel operation.
centers = repmat(W,[numel(gray),1]);
disp(W);%%Weight Vectors
distance = ((vector-centers).^2);% Find distance between center and pixel value.
%[~,lb] = min(distance,[],2);% Choose cluster index of minimum distance.
%lb = reshape(lb,size(gray));% Reshape the labelled index vector.
img= double(bw);
clusterNum=size(distance,2);
[Unow, center, now_obj_fcn ] = FCMforImage( img, clusterNum );
for i=1:clusterNum
    %subplot(2,2,i+1);
    %imshow(Unow(:,:,i),[]);
end
x=i;
y=imcomplement(Unow(:,:,x));
imshow(y);title('MMTD FCM');

%%
%%%%%%%ROLLING BALL METHOD%%%%%%%%%%

%%For instance the Rolling-ball algorithm uses a ball as a structuring element and performs the top-hat transform[2].
%REF:Internet
%se = strel('disk',12);
se = strel('ball',10,5);%ROLLING BALL ELEMENT
y= imtophat(y,se);
imshow(y);title('LUNG BORDER CORRECTION');
glcms = graycomatrix(y);
stats = graycoprops(glcms);
c=stats.Contrast;
co=stats.Correlation;
e=stats. Energy;
h=stats.Homogeneity;
fet=[c;co;e;h];
%feature=cat(2,feature,featureVec);%
%feature=feature';
%save feature.mat feature
out=svmclassify(SVMstruct,fet');
if out==1
    clc;
    msgbox('Tumour Lung')
else
    clc;
    msgbox('Non Tumour Lung')
end
%%