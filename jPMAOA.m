function [sFeat,Sf,Nf,curve,ErrorCurve] = jPMAOA(feat,label,N,max_FEs,HO)

%% Problem Definition

% Define the fitness function of wrapper-based feature selection approach
% KNN classifier is used to evaluate the accuracy level
fun = @jFitnessFunction;

% To obtain total number of original feature (Dimension)
Dim = size(feat,2);
nVar = Dim;
nPop = N;
lb = 0; %lower boundary
ub = 1; %upper boundary
VarMin = lb;
VarMax = ub;
thres = 0.5; % a threshold value used to convert the real-value into binary
MaxFE = max_FEs;

VarSize = [1 nVar]; % Unknown Variables Matrix Size

%% Mutation Parameters

K = max(3, round(0.30 * nVar));   % 30% of dimensions, at least 3
DL = 10; %parameter of mutation-restarting phase

% AOA Constants
Mu      = 0.499;
Alpha   = 5;
MOA_Min = 0.2;
MOA_Max = 1;

% Counter
fes = 0;            % Fitness evaluation counter
G = 0;
fes2 = 1;

%% Initialization
% Empty Structure for Individuals
empty_individual.Position = [];
empty_individual.Cost = [];

%% Preallocate
curve      = zeros(1,MaxFE);
ErrorCurve = zeros(1,MaxFE);
Leader   = zeros(floor(MaxFE/nPop)+DL, Dim);

% Initialize Population Array
pop = repmat(empty_individual, nPop, 1);

% Initialize Best Solution
BestSol.Cost = inf;

% Initialize Population Members
for i=1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    %Evaluation
    [pop(i).Cost, pop(i).Error] = fun(feat, label, pop(i).Position, HO, thres); % Evaluation
    fes = fes + 1;
    if pop(i).Cost < BestSol.Cost
        BestSol = pop(i);
    end
end

%% Main Loop

tic;

while fes <= MaxFE


    %% Peer Guided Stage (peer difference + Linear Decay)
    newsol = repmat(empty_individual, nPop, 1);

    % AOA parameters for this phase
    MOP = 1 - (fes^(1/Alpha) / MaxFE^(1/Alpha));
    MOA = MOA_Min + fes * ((MOA_Max - MOA_Min) / MaxFE);
    % Linear Decay
    AOA_fac = ((VarMax- VarMin) * Mu + VarMin) * (1 - fes/MaxFE);

    for i = 1:nPop

        % Learn from randomly selected peer
        A = 1:nPop;  A(i) = [];
        j = A(randi(nPop-1));

        % if the partner is better than the current solution, then it will
        % move towards the better solution
        if pop(j).Cost < pop(i).Cost
            Step = pop(j).Position - pop(i).Position; %
        else
            Step = pop(i).Position - pop(j).Position;
        end
        
        r1 = rand(); r2 = rand(); r3 = rand();

        if r1 < MOA
            if r2 > 0.5
                newsol(i).Position = (pop(i).Position+ (rand*Step))/(MOP+eps) * AOA_fac;
            else
                newsol(i).Position = (pop(i).Position+ (rand*Step)) * MOP * AOA_fac;
            end
        else
            if r3 > 0.5
                newsol(i).Position = (pop(i).Position + (rand*Step)) - MOP * AOA_fac;
            else
                newsol(i).Position= (pop(i).Position + (rand*Step)) + MOP * AOA_fac;
            end
        end

        % Dynamic Boundary Constraint - Eq. (4)
        for d = 1:nVar
            if (newsol(i).Position(d) < VarMin) || (newsol(i).Position(d) > VarMax)
                newsol(i).Position(d) = VarMin + rand * (VarMax - VarMin);
            end
        end

        % Evaluation
        [newsol(i).Cost, newsol(i).Error] = fun(feat, label, newsol(i).Position, HO, thres);
        fes = fes + 1;

        if newsol(i).Cost < pop(i).Cost
            pop(i) = newsol(i);
            if pop(i).Cost < BestSol.Cost
                BestSol = pop(i);
            end
        end

        % Update convergence
        curve(fes2:fes) = BestSol.Cost;
        ErrorCurve(fes2:fes) = BestSol.Error;
        fes2 = fes + 1;
        if fes >= MaxFE, break; end
    end

    if fes > MaxFE
        break;
    end

    G = G + 1; % Update generation counter
    Leader(G,:) = BestSol.Position;
    %% Mutation-Restarting Phase
    newsol = repmat(empty_individual,nPop,1);

    for i = 1:nPop

        if mod(G,2)~=0 %G is an odd number then Mutation

            %Learn from randomly selected peer
            A = 1:nPop;
            %Eliminate the index of learner itself
            A(i)=[];
            %Randomly select the indices X1 and X2 of two learners
            Rand_Indices = randperm(numel(A),2);
            X1 = A(Rand_Indices(1));
            X2 = A(Rand_Indices(2));

            % if counter G is larger than the delay parameter
            % then calculate new solution based on the two learners and
            % best solution
            if G > DL

                Leader_DL = Leader(G-DL,:);
                %Calculate new solution for learner i
                newsol(i).Position = ...
                    pop(i).Position + (2*rand-1)*(Leader(G,:)-Leader_DL) ...
                    +rand.*(pop(X1).Position-pop(X2).Position);
            else

                %Calculate new solution for learner i
                newsol(i).Position = pop(i).Position + ...
                    rand.*(pop(X1).Position-pop(X2).Position);
            end

        else %G is an even number then do Restarting

            %Copy the current position of learner i into new position
            newsol(i).Position = pop(i).Position;

            %Randomly select K = 30% dimension to restart
            index = randperm(nVar,K);

            %Restarting strategy on the randomly selected dimension
            for d=1:K
                d_index = index(d);
                newsol(i).Position(d_index) = VarMin + rand*(VarMax-VarMin);
            end

        end

        %Dynamic Boundary Constraint - Eq. (7)
        for d=1:nVar
            if (newsol(i).Position(d)<VarMin) || (newsol(i).Position(d)>VarMax)

                %Initialize the solution that violate the boundary
                %constraint
                newsol(i).Position(d)= VarMin + rand*(VarMax-VarMin);

            end
        end

        %Evaluation
        [newsol(i).Cost, newsol(i).Error] = fun(feat, label, newsol(i).Position, HO, thres); % Evaluation
        fes = fes+1;

        if newsol(i).Cost < pop(i).Cost
            pop(i) = newsol(i);
            %Update so far best solution if the updated learner is
            %better
            if pop(i).Cost < BestSol.Cost
                BestSol = pop(i);
            end
        end

        %Update the best so far value after each fitness evaluation
        curve(fes2:fes) = BestSol.Cost;
        ErrorCurve(fes2:fes) = BestSol.Error;
        fes2 = fes + 1;
    end

    %Merge current and new population
    Pop_Merge = [pop; newsol];

    %Perform sorting on merged population based on fitness
    [~, Sorted_Indices] = sort([Pop_Merge.Cost]);

    %clone one population to capture all original values
    clone = Pop_Merge;

    %Rearrange all learners in pop from best to worst
    pop = clone(Sorted_Indices(1:nPop));
    G=G+1; %Update generation counter

end

Select_Index = find(BestSol.Position > thres);
NotSelect_Index = find(BestSol.Position <= thres);
BestSol.Position(Select_Index) = 1;
BestSol.Position(NotSelect_Index) = 0;
Sf    = Select_Index;
Nf    = length(Sf);
sFeat = feat(:,Sf);

end
