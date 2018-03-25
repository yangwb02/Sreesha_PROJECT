clc;
clear all;
close all;
%function [result] = Splat_Feature_SVM(feat1)
%training of svm
a=load('Trainset.mat');%load the trainingset
TrainingSet=a.sample;
b=load('class.mat');%lod the known classes
GroupTrain=b.groupTrain;
Testset=TrainingSet(1,:);%load the test set
u=unique(GroupTrain);
numclasses=length(u);
result=zeros(length(Testset(:,1)),1);

for k=1:numclasses
    Glvall=(GroupTrain==u(k));
    models(k)=svmtrain(TrainingSet,Glvall);
end
%svm testing
for j=1:size(Testset,1)
    for k=1:numclasses
        if(svmclassify(models(k),Testset(j,:)))
            break;
        end
    end
    result(j)=k;% classifier result
end