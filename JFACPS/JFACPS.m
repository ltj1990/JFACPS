function [acc,acc_ite,max_acc,A,func,Zs,Zt]= JFACPS(Xs,Ys,Xt,RealYt,options)

acc_ite=[];
func=[];
%% Parameters setting
dim=options.dim;
alpha=options.alpha;
beta=options.beta;
lambda = options.lambda;
ker = options.kernel_type;
gamma = options.gamma;
num_iter = options.T;
sC=options.sC;
eta=options.eta;
t1=options.t1;
t2=options.t2;

%% Initialization
[X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt,1);
if (~strcmpi(ker,'primal')) && ker~=0
    X=kernelProject(ker,X,[],gamma);
    Xs=X(:,1:ns);
    Xt=X(:,ns+1:end);
    m=n;
end
% Init target pseudo labels
Yshot=hotmatrix(Ys,C,0);
% Ytpseudo=classifySVM(Xs,Ys,Xt);
options.eta =eta;
options.t1 =t1;
options.t2 = t2;
Ytpseudo = MCPSVM(Xs', Ys, Xt', options);
    
predLabels=Ytpseudo;
acc=getAcc(Ytpseudo,RealYt);
fprintf('init acc:%.4f\n',acc);
YsHotSameWeight=hotmatrix(Ys,C,1);
% Init U_s * G_s by Eq.(8)
Fs=Xs*YsHotSameWeight; % m * c
cWeight=zeros(1,C);
maxC=inf;
for i=1:C
    cWeight(i)=1/(ns-length(find(Ys==i)));
    maxC=min(maxC,length(find(Ys==i)));
end
% Init \hat{U}_s * G_s by Eq.(8)
cWeight=ones(ns,1)*cWeight;
YsHotDiffWeight=((1-Yshot).*cWeight);
Frs=Xs*YsHotDiffWeight;
% Set default value
Sw=0;
Sw2=0;
max_acc = acc;
for iter = 1:num_iter
    if iter>1
        % Solve Eqs.(8)
        tmpT=Xt-Fs*probYt';
        Sw=tmpT*tmpT';
        Xtc=Xt(:,logical(trustable));
        % Solve Eqs.(10)-(11)
        hotYt=hotmatrix(predLabels(logical(trustable)),C,0);
        tmpT2=Xtc-Frs*hotYt';
        Sw2=tmpT2*tmpT2';
    end
    % Solve Eq.(5)
    if iter>1
        % Use [Xs,X_{t,tr}]
        Ymain=[Ys;predLabels(logical(trustable))];
        XL=[Xs,Xt(:,logical(trustable))];
    else
       % Use Xs in 1st iteration
       Ymain=Ys;
       XL=Xs;
    end
    XL= L2Norm(XL')';
    manifold.Metric='Cosine';
    manifold.WeightMode='Binary';
    manifold.NeighborMode='Supervised';%'Supervised';
    manifold.gnd=Ymain;
    manifold.normr=1;
    manifold.k=0;
    [Ls,D,~]=computeL(XL,manifold);
    Ls=XL*Ls*XL';
    Ds=XL*D*XL';
    % Solve Eq.(12)
    [A,~]=eigs(Ls+lambda*eye(m)+alpha*(Sw),beta*Sw2+1e-6*Ds,dim,'sm');
    Zs=A'*Xs;
    Zt=A'*Xt;
    % Solve DPL and get probability
    pos=C-sC+1;
    p=1-(iter/num_iter);
    if i>1
       lastPredLabels= predLabels;
    else
        lastPredLabels=[];
    end
    [probYt,trustable,predLabels] = getDPL(Zs,Ys,Zt,predLabels,lastPredLabels,pos,p);
    % calculate ACC
        func(iter)=trace(Ls+lambda*eye(m)+alpha*(Sw))/trace(beta*Sw2+1e-6*Ds);
    acc=getAcc(predLabels,RealYt);
    acc_ite(iter)=acc;
    if  acc >= max_acc
        max_acc =  acc;
    else
        return
    end
    fprintf('Iteration=%d, Acc:%0.3f\n', iter, acc);
end
end