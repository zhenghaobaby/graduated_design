function TempotronClassify()
% Tempotron: a neuron that learns spike timing-based decisions
% Rober Gutig 2006 Nature Neuroscience
clear; clc;
NumImages=26;
for i=1:NumImages
    ImageName=strcat('C:\Users\Administrator\Desktop\Icon16X16\Letter-',char('A'+i-1),'-black-icon');% 从icon16X16文件夹中读取所有图片
    ImageMatrix=imread(ImageName,'bmp');% 读取图片为灰度图，保存在矩阵中，取反
    ImageMatrix=~ImageMatrix;  % make the white pixel be 0, and black be 1;
    TrainPtns(:,i)=image2ptn(ImageMatrix);
end
TrainPtns=TrainPtns*1e-3;  % scale to ms
nAfferents = size(TrainPtns,1);
nPtns = NumImages;
%nOutputs = 1;    %%%%%%%%%%   1
nOutputs = 5;
 
loadData=0;% 是否载入已保存的模型
 
V_thr = 1; V_rest = 0;
T = 256e-3;         % pattern duration ms
dt = 1e-3;
tau_m = 90.63e-3;
%tau_m = 20e-3; % tau_m = 15e-3;？？？
tau_s = 7.73e-3;
%tau_s = tau_m/4;
% K(t?ti)=V0(exp[?(t?ti)/τ]–exp[?(t?ti)/τs])
aa = exp(-(0:dt:3*tau_m)/tau_m)-exp(-(0:dt:3*tau_m)/tau_s);
 
V0 = 1/max(exp(-(0:dt:3*tau_m)/tau_m)-exp(-(0:dt:3*tau_m)/tau_s));
lmd = 2e-2;%1e-2/V0;   % optimal performance lmd=3e-3*T/(tau_m*nAfferents*V0)  1e-4/V0;
maxEpoch = 100;
mu = 0.99;  % momentum factor
% generate patterns (each pattern consists one spik-e per afferent)
 
if loadData ==0 %初始化网络
    weights = 1e-2*randn(nAfferents,nOutputs);  % 1e-3*randn(nAfferents,1);
    save('weights0','weights');
else
    load('weights0','weights');
end
%Class = logical(eye(nOutputs));     % desired class label for each pattern
%Class = false(1,26); Class(26)=true;
Class = de2bi(1:26,'left-msb'); Class=Class';
 
correctRate=zeros(1,maxEpoch);
dw_Past=zeros(nAfferents,nPtns,nOutputs);  % momentum for accelerating learning.上一个权重的更新，用于动量计算
for epoch=1:maxEpoch    
    Class_Tr = false(nOutputs,nPtns);  % actual outputs of training
    for pp=1:nPtns 
 %       Class_Tr = false(nOutputs,1);  % actual outputs of training
                
        for neuron=1:nOutputs
            Vmax=0; tmax=0;
            fired=false;        
            Vm1=zeros(1,256); indx1= 1; % trace pattern 1
            for t=dt:dt:T
                Vm = 0; 
                if fired==false
                    Tsyn=find(TrainPtns(:,pp)<=t+0.1*dt);    % no cut window
                else
                    Tsyn=find(TrainPtns(:,pp)<=t_fire+0.1*dt); % shut down inputs
                end
                if ~isempty(Tsyn)                    
                    A1=TrainPtns(:,pp);
                    A2=A1(Tsyn);
                    %K = 0.777*exp(-(t-A2)*1000/16.8);
                    %K = (-0.346*log((t-A2)*1000+1)+2.0708)/2.0708;
                    K =myNeuralNetworkFunction((t-A2)*1000);
                    %K =network.network3((t-A2)'*1000);
                    %K = K';
                    %K =V0*(exp(-(t-A2)/tau_m)-exp(-(t-A2)/tau_s)); % the kernel value for each fired afferent
                    A1=weights(:,neuron);
                    firedWeights=A1(Tsyn);
                    Vm = Vm + firedWeights'*K ;
                    %Vm = Vm + K*firedWeights;
                end
 
                Vm = Vm + V_rest;
                if Vm>=V_thr && fired==false % fire
                    fired=true;
                    t_fire=t;
                    Class_Tr(neuron,pp)=true;
                end
                if Vm>Vmax
                    Vmax=Vm; tmax=t;
                end
 
                %if pp==1
                    Vm1(indx1)=Vm;
                    indx1=indx1+1;
                %end
            end
 
            %if pp==1
                figure(1); plot(dt:dt:T,Vm1);
                title(strcat('Image ',char('A'+pp-1),'; neuron: ',num2str(neuron))); drawnow;
            %end
            if Vmax<=0
                tmax=max(TrainPtns(:,pp));
            end
            
            if Class_Tr(neuron,pp)~=Class(neuron,pp) % error
                
                Tsyn=find(TrainPtns(:,pp)<=tmax+0.1*dt); 
                if ~isempty(Tsyn)                    
                    A1=TrainPtns(:,pp);
                    A2=A1(Tsyn);
                    %K = 0.777*exp(-(t-A2)*1000/16.8);
                    %K = (-0.346*log((t-A2)*1000+1)+2.0708)/2.0708;
                    K = myNeuralNetworkFunction((t-A2)*1000);
                    %K =network.network3((t-A2)'*1000);
                    %K = K';
                    %K =V0*(exp(-(tmax-A2)/tau_m)-exp(-(tmax-A2)/tau_s)); % the kernel value for each fired afferent
                    A1=weights(:,neuron);
                    dwPst=dw_Past(:,pp,neuron);
                    if fired==false    % LTP
                        Dw=lmd*K;
                    else           % LTD 
                        Dw=-1.1*lmd*K;
                    end
                    A1(Tsyn) = A1(Tsyn) + Dw + mu*dwPst(Tsyn);
                    weights(:,neuron)=A1;
                    dwPst(Tsyn)=Dw;
                    dw_Past(:,pp,neuron) = dwPst;
                end                
            end            
            
        end  % end of one neuron computation
        
   end % end of one image
   Class_use = bi2de(Class', 'left-msb');
   Class_Tr_use = bi2de(Class_Tr','left-msb');
   CC= (Class_use==Class_Tr_use);
   a = sum(CC)/length(CC);
   
   correctRate(epoch)=mean(a);

   %CC = bi2de(Class_Tr','left-msb');
end
save('TrainedWt','weights');
figure(2); plot(1:maxEpoch,correctRate,'-b.');
xlabel('epochs');
ylabel('accuracy');
end
%%将图片编码为脉冲序列并保存在向量中
function spikeTrain=image2ptn(A)  
%% convert a image to a spike train saved in a vector
RandParts=1;
% A1=A';
% B=[A1(:);A(:)];
% numPixels=length(B);
% numInputNeurons=numPixels/8; % 64 neurons
% spikeTrain=zeros(numInputNeurons,1);
% for i=1:numInputNeurons
%     Bits=B((1+(i-1)*8):(8+(i-1)*8));
%     Bits=Bits';
%     spikeTime=bi2de(Bits,'left-msb');    
%     if spikeTime==0
%         spikeTime=2^8;  % put 0 to the end of the interval
%     end
%     spikeTrain(i)=spikeTime;
% end
spikeTrain=zeros(32,1);
if RandParts==1
    loadR=1;
    AR=A(:);
    if loadR==0
        R = randperm(size(A,1)*size(A,2));
        save('RandIndex','R');
    else
        load('RandIndex','R');
    end
    numRandNeus=32;
    for i=1:numRandNeus
        IndexR=R((1+(i-1)*8):(8+(i-1)*8));
        Bits=AR(IndexR);
        Bits=Bits';
        spikeTime=bi2de(Bits,'left-msb');    % 二进制转十进制
        if spikeTime==0
            spikeTime=2^8;  % put 0 to the end of the interval
        end
        spikeTrain(i)=spikeTime;
    end
end
end

