% This function does projection onto the non-convex constraint set

function coeff_new = ADMM_proj_binary(coeff, r, theta, threshold)

n = size(coeff, 1);

x = coeff(:,1);
y = coeff(:,2);
z = coeff(:,3);

% note different cells can have different threshold here, corresponding to
% diffenrent delta mass at 0


% percent we will set to be zero
% size(x(x < threshold),1) / n;


% do threshold before doing the projection step
% this is like doing a coordinate descent step
% x(x < threshold) = 0;
% y(x < threshold) = 0;
% z(x < threshold) = 0;
count = 0;

% we always do projection before we do threshold
% since the solution from the first step has no constrains at all on the
% coefficients and using x along does not work well
for i = 1 : n
    
    if (abs(atan(z(i)/y(i))) <= theta)  % if we are within the sector
        theta_proj = atan(z(i)/y(i));
    elseif (z(i) > 0)                   % otherwise the projection will be at one of the endpoints of the arc
        theta_proj = theta;
    elseif (z(i) < 0)
        theta_proj = -theta;
    elseif (z(i) == 0)                  % when z(i) = 0, we pick either endpoint at random
        if (y(i) == 0)
            theta_proj = unifrnd(-theta,theta);
        else
            if rand <= 0.5
                theta_proj = theta;
            else
                theta_proj = -theta;
            end
        end
    end
    % now we project onto the line specified by [1/r cos(theta_proj) sin(theta_proj)]
    unit_vec = [1/r cos(theta_proj) sin(theta_proj)]'/norm([1/r cos(theta_proj) sin(theta_proj)]');
    projection_length = dot([x(i) y(i) z(i)]', unit_vec);
    
    
    % removing the following line causes the program to behave super
    % badly......
    % so using x along before projection is probably not a good idea...
    
    x(i) = projection_length * unit_vec(1);
    
    
    
    
    % y(i) and z(i) will be determined later
    %y(i) = projection_length * unit_vec(2);
    %z(i) = projection_length * unit_vec(3);
    
    % if we do threshold we always want to project the remaining ones to 1
    % doesn't seem to be working well....
    %         x(i) = 1;
    %         y(i) = r * cos(theta_proj) * x(i);
    %         z(i) = r * sin(theta_proj) * x(i);
    
    % note 0.5 here has nothing to do with threshold here
    % it comes from the binary thing and = (1+0)/2
    if x(i) < threshold          % do projection onto the truncated cone
        count = count + 1;
        x(i) = 0;
        y(i) = 0;
        z(i) = 0;
    else                    % else we project it to 1
        x(i) = 1;
        y(i) = r * cos(theta_proj) * x(i);
        z(i) = r * sin(theta_proj) * x(i);
    end
    
end

count / n

coeff_new = [x y z];


