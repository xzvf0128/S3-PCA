function [PC] = S3_PCA(data,num_PC,k,labels)
% 
[M,N,B]=size(data);
Results_segment= seg_im_class(data,labels);
Num=size(Results_segment.Y,2);
for i=1:Num
    Results_segment.Y{1,i} = findConstruct(Results_segment.Y{1,i},Results_segment.cor{1,i},k);
    [P] = Eigenface_f(Results_segment.Y{1,i}',num_PC);
    PC = (Results_segment.Y{1,i})*P;
    X(Results_segment.index{1,i},:)=PC;
    A(Results_segment.index{1,i},:)=Results_segment.Y{1,i};
end
[P] = Eigenface_f(A',num_PC);
A_PC=A*P;
X = [X,A_PC];
[P] = Eigenface_f(X',num_PC);
X = X*P;
PC = reshape(X,M,N,num_PC);