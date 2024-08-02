function [Cn,Xn]=centroid_align(x,str)
    % Align the original data covariances by congruent transform
    % 用同余变换将原始数据的协方差对齐
    % Input:
    %   x: the original data covariances K*T*N
    %   str: congruent transform by Riemanian or Euclidean mean
    % Output:
    %   Cn: centralized covariance matrices K*K*N
    %   Xn: centralized raw data K*T*N

    % Author: Wen Zhang and Dongrui Wu
    % Date: Oct. 9, 2019
    % E-mail: wenz@hust.edu.cn

    tmp_cov=zeros(size(x,1),size(x,1),size(x,3));
    for i=1:size(x,3)
        tmp_cov(:,:,i)=cov(x(:,:,i)');
    end

    C = mean_covariances(tmp_cov,str);  % should include the covariancetoolbox 
    P = C^(-1/2);
    
    Cn=zeros(size(x,1),size(x,1),size(x,3));
    for j=1:size(x,3)
        Cn(:,:,j)=P*squeeze(tmp_cov(:,:,j))*P;  %  squeeze：除去size为1的维度
    end

    Xn=zeros(size(x));
    for i=1:size(x,3)
        Xn(:,:,i)=P*x(:,:,i);
    end
end
