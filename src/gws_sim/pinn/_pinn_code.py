import pandas as pd
import deepxde as dde
import numpy as np
import sys
import os
import math
import torch
import re
import scipy as sp
from scipy.integrate import odeint
import ast

print("step a")
print(len(sys.argv))
if len(sys.argv) != 14:
    raise Exception("Usage: python script.py <dataframe_file.csv> <string_list_file.txt> <t_start> <t_end> <initial_state>")

dataframe_file = sys.argv[1]
string_list_file_equations = sys.argv[2]
string_list_file_params = sys.argv[3]
t_start = sys.argv[4]
t_end = sys.argv[5]
string_list_file_initial_state = sys.argv[6]
number_hidden_layers = sys.argv[7]
width_hidden_layers = sys.argv[8]
number_iterations = sys.argv[9]
string_file_additional_functions = sys.argv[10]
number_iterations_predictive_controller = sys.argv[11]
control_horizon = sys.argv[12]
simulator_type = sys.argv[13]

t_start_ = float(t_start)
t_end_ = float(t_end)
control_horizon = float(control_horizon)

if not os.path.exists(dataframe_file):
    raise Exception(f"Error: DataFrame file '{dataframe_file}' not found.")

try:
    df = pd.read_csv(dataframe_file)
except Exception as err:
    raise Exception(f"Error: DataFrame file '{dataframe_file}' not found.\n{err}")

# Read list of strings from text file
try:
    with open(string_list_file_equations, 'r') as f:
        string_list_equations = [line.strip() for line in f.readlines()]
except FileNotFoundError as err:
    raise Exception(f"Error: String list file '{string_list_file_equations}' not found.\n{err}")

# Read list of strings from text file
try:
    with open(string_list_file_params, 'r') as f:
        string_list_params = [line.strip() for line in f.readlines()]
except FileNotFoundError as err:
    raise Exception(f"Error: String list file '{string_list_file_params}' not found.\n{err}")

# Read list of strings from text file
try:
    with open(string_file_additional_functions, 'r') as f:
        string_additional_functions = f.read()
except FileNotFoundError as err:
    raise Exception(f"Error: String_additional_functions '{string_file_additional_functions}' not found.\n{err}")

# Read list of strings from text file
try:
    with open(string_list_file_initial_state, 'r') as f:
        string_list_initial_state = [line.strip() for line in f.readlines()]
except FileNotFoundError as err:
    raise Exception(f"Error: String list file '{string_list_file_initial_state}' not found.\n{err}")

##############################################################################
C = []
ic = []
observe_y = []
data_list = []
temp = []

neural_network_weights = np.ones(3*len(string_list_equations))

# Split the code into lines and extract the first line
first_line = string_additional_functions.strip().split('\n')[0]

for i in range(len(string_list_params)):
    C.append(dde.Variable(float(string_list_params[i])))
    print(C)

# Count the number of variables in the first line
num_vars = len(first_line.split('=')[0].strip().strip('[]').split(','))

# Ensure the list C has the same number of elements
assert len(C) == num_vars, f"The list C must contain {num_vars} elements."

# Execute the first line to assign values
exec(first_line)

# Extract the variable names from the first line
params_names = first_line.split('=')[0].strip().strip('[]').split(',')
params_names = [param.replace(" ", "") for param in params_names]

# Assign the variables to a new list
params_list = [eval(var.strip()) for var in params_names]

exec(string_additional_functions) # the right place of this line is here !!!

external_trainable_variables = C

###################################################################################

def _predict_substrate(horizon, F_in, t0, X0, S0, P0, K):
    '''
    Use model to predict substrate level at horizon time.

    Args:
        horizon: horizon control time [hours]
        F_in: inlet mass flow [L/hour]
    Returns:
        Sf = predicted subtrate concentration at  [g/L]
    '''

    S_in = 350

    print('__________________________________________________')
    print("start _predict_substrate")

    for i in range(len(params_list)):
        exec(f"{params_names[i]} = {K[i]}")

    t_ = t0

    y = [X0, S0, P0]

    D = F_in
    while t_ < t0 + horizon:

        y[0] = y[0] + dt_predict * eval(string_list_equations[0])
        y[1] = y[1] + dt_predict * eval(string_list_equations[1])
        y[2] = y[2] + dt_predict * eval(string_list_equations[2])

        t_old = t_

        t_ = t_ + dt

        print(f'{t_old} + {dt} = {t_} ---> {t0+horizon}')

    print(f"end _predict_substrate : predicted subsrate = {y[1]}")
    print('__________________________________________________')

    return y[1]

def step_model(D, X_, S_, P_, dt, t_, K):

    S_in = 350

    for i in range(len(params_list)):
        exec(f"{params_names[i]} = {K[i]}")

    equations_list = []

    y = [X_, S_, P_]

    equations_list.append(y[0] + dt * eval(string_list_equations[0]))
    equations_list.append(y[1] + dt * eval(string_list_equations[1]))
    equations_list.append(y[2] + dt * eval(string_list_equations[2]))

    return equations_list

def step_controller(S_setpoint, X_meas, S_meas, P_meas, t, S_in, horizon, K):
    '''
    Args:
        horizon: horizon control time [hours]
        S_setpoint: desired subtrate concentration at horizon time [g/L]
        X_meas, S_meas, E_meas: biomass, substrate and ethanol concentration [g/L]
    Returns:
        D_in: control value
    '''

    '''
    self.model.X = X_meas
    self.model.S = S_meas
    self.model.P = E_meas
    self.model.S_in = S_in
    '''

    # bisective method:
    # Sf is a function of D
    # find the zero of:     Sf(D) - S_setpoint = 0

    t0 = t
    X0 = X_meas
    S0 = S_meas
    P0 = P_meas

    def f(D): return _predict_substrate(horizon, D, t0, X0, S0, P0, K) - S_setpoint
    a = D_min
    b = D_max
    f_a = f(a)
    print(f'f(a) = {f_a}')
    print('__________')
    f_b = f(b)
    print(f'f(b) = {f_b}')
    print('__________')

    if f_a > 0:
        print('f(a) is positive')
        print('end: controller')
        return a
    elif f_b < 0:
        print('f(b) is negative')
        print('end: controller')
        return b

    iteration_controller = 0
    while iteration_controller < iter_max:
        iteration_controller += 1
        print(f'iteration_controller = {iteration_controller}')
        m = (a + b) / 2
        f_m = f(m)
        print(f'f(m) = {f_m}')
        print('__________')
        if f_m * f_a < 0:
            b = m
            f_b = f_m
        else:
            a = m
            f_a = f_m

    return m

def uniform_step_function_generator(values_list:list, time:list):
    Func = 0
    i = 0
    subdivision_length = max(time) / len(values_list)
    mult = 999999999999999999
    for val in values_list:
        Func = Func + val * ((1+np.tanh(mult*(time - i * subdivision_length)))-(1+np.tanh(mult*(time - (i+1) * subdivision_length)))) * 0.5
        i += 1

    return Func

###############################

def read_last_line_and_clean(file):
    with open(file, 'rb') as f:
        # Go to the end of the file
        f.seek(0, 2)
        # Read backwards until a newline character is found
        position = f.tell()
        while position >= 0:
            f.seek(position)
            char = f.read(1)
            if char == b'\n' and position != f.tell() - 1:
                last_line = f.readline().decode('utf-8')
                break
            position -= 1
        else:
            f.seek(0)
            last_line = f.readline().decode('utf-8')  # If the file has no newline character

    # Remove characters before the first '['
    index_bracket = last_line.find('[')
    if index_bracket != -1:
        last_line = last_line[index_bracket:]
    else:
        last_line = ''

    return last_line

def convert_string_to_list(string):
    try:
        # Use literal_eval to convert the string to a list
        lst = ast.literal_eval(string)
        if isinstance(lst, list):
            return lst
        else:
            raise ValueError("The string is not a valid list.")
    except (ValueError, SyntaxError) as e:
        print("Conversion error: ", e)
        return []

###################################################################################

print(f"type of string_list_equations : {type(string_list_equations)}, number of elements = {len(string_list_equations)}")
print(string_list_equations)
print(f"type of string_list_params : {type(string_list_params)}")
print(string_list_params)
print(f"type of string_list_initial_state : {type(string_list_initial_state)}")
print(string_list_initial_state)
print(f"type of string_list_initial_state[0] : {type(string_list_initial_state[0])}")
print(f"type of float(string_list_initial_state[0]) : {type(float(string_list_initial_state[0]))}")

print(f"type of additional_functions : {type(string_additional_functions)}")
print(string_additional_functions)

print(f"simulator_type = {simulator_type}")

if simulator_type == "PINN":

    variable = dde.callbacks.VariableValue(
        external_trainable_variables, period=1000 #, filename="variables.dat"
    )

    # Most backends
    def ODE_model(x, y_):

        equations_list = []
        dy_x = []
        y = []

        for j in range(len(string_list_equations)):
            y.append(y_[:, j:j+1])
            dy_x.append(dde.grad.jacobian(y_, x, i=j))
            #print(f"dy_x = {dy_x}")

        for i in range(len(string_list_equations)):
            equations_list.append(dy_x[i] - eval(string_list_equations[i]))

        return equations_list

    def boundary(_, on_initial):
        return on_initial

    geom = dde.geometry.TimeDomain(t_start_, t_end_)

    df_data = df
    data_set_array = df.to_numpy()

    observe_t = data_set_array[:, 0:1]
    print(observe_t)
    for k in range(len(string_list_equations)):

        # Initial conditions
        ic.append(dde.icbc.IC(geom, lambda X: float(string_list_initial_state[k]), boundary, component=k))
        print(f"float(string_list_initial_state[{k}]) = {float(string_list_initial_state[k])}")
        # Get the train data
        observe_y.append(dde.icbc.PointSetBC(observe_t, data_set_array[:, k+1:k+2], component=k))
        print(data_set_array[:, k+1:k+2])
        print("step 1")

    data_list = ic + observe_y
    print(data_list)
    data = dde.data.PDE(
        geom,
        ODE_model,
        data_list,
        num_domain=400,
        num_boundary=len(string_list_equations),
        anchors=observe_t,
    )
    print("step 2")

    net = dde.nn.FNN([1] + [int(width_hidden_layers)] * int(number_hidden_layers) + [len(string_list_equations)], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    print("step 3")

    # train adam
    model.compile(
        "adam", lr=0.001, external_trainable_variables=external_trainable_variables, loss_weights=[1,1,1 , 1,1,1, 1,1,1]
    )
    losshistory, train_state = model.train(iterations=int(number_iterations), callbacks=[variable])
    print("step 4")

    # train lbfgs
    model.compile("L-BFGS", external_trainable_variables=external_trainable_variables)
    losshistory, train_state = model.train(callbacks=[variable])

    idx = np.argsort(train_state.X_test[:, 0])

    t = train_state.X_test[idx, 0]
    best_y = train_state.best_y

    LIST = best_y[idx, :]

    dft = pd.DataFrame(t, columns=['time'])
    dfy = pd.DataFrame(LIST)

    new_column_names_result = [f'result_col_{i+1}' for i in range(len(dfy.columns))]
    dfy.columns = new_column_names_result

    new_column_names_data = [f'data_col_{i+1}' for i in range(len(df_data.columns))]
    df_data.columns = new_column_names_data

    loss_steps = losshistory.steps
    loss_train = np.array([np.sum(loss) for loss in losshistory.loss_train])
    loss_test = np.array([np.sum(loss) for loss in losshistory.loss_test])

    df_loss_steps = pd.DataFrame(loss_steps, columns=['loss_steps'])
    df_loss_train = pd.DataFrame(loss_train, columns=['loss_train'])
    df_loss_test = pd.DataFrame(loss_test, columns=['loss_test'])

    print("step 5")
    df_result = pd.concat([dft, dfy, df_data, df_loss_steps, df_loss_test, df_loss_train], axis=1)
    print("step 6")
    df_result.to_csv("pinn_result.csv", index=False)
    print("step 7")

else:
    D_min, D_max = 0, 1  # control value bounds
    dt_predict = 5e-3  # model integration time step (euler)
    iter_max = 15  # for bisective algorithm

    # time points
    maxtime = int(t_end_) - int(t_start_)
    time = np.linspace(int(t_start_), maxtime, int(t_end_))

    dt = 0.1

    u = 0.0001 * time #((1+np.tanh(time-40))*(1-np.tanh(time-60)))

    X0 = float(string_list_initial_state[0])
    S0 = float(string_list_initial_state[1])
    P0 = float(string_list_initial_state[2])
    K = C

    for iteration in range(int(number_iterations_predictive_controller)):

        K = [float(num) for num in K]

        for i in range(len(params_list)):
            exec(f"{params_names[i]} = dde.Variable(({K[i]})**(1/2))")

        ex_input = u # exogenous input

        # interpolate time / lift vectors (for using exogenous variable without fixed time stamps)
        def ex_func(t):
            spline = sp.interpolate.Rbf(
                time, ex_input, function="thin_plate", smooth=0, episilon=0
            )
            return spline(t)

        def ODE_model(x, y_):

            equations_list = []
            dy_x = []
            y = []

            for j in range(len(string_list_equations)):
                y.append(y_[:, j:j+1])
                dy_x.append(dde.grad.jacobian(y_, x, i=j))
                #print(f"dy_x = {dy_x}")

            for i in range(len(string_list_equations)):
                equations_list.append(dy_x[i] - eval(string_list_equations[i]))

            return equations_list

        def boundary(_, on_initial):
            return on_initial

        geom = dde.geometry.TimeDomain(t_start_, t_end_)

        # Initial conditions
        ic1 = dde.icbc.IC(geom, lambda X: X0, boundary, component=0)
        ic2 = dde.icbc.IC(geom, lambda X: S0, boundary, component=1)
        ic3 = dde.icbc.IC(geom, lambda X: P0, boundary, component=2)

        # Get the train data
        df_data = df
        data_set_array = df.to_numpy()

        observe_t = data_set_array[:, 0:1]

        for k in range(len(string_list_equations)):

            # Initial conditions
            ic.append(dde.icbc.IC(geom, lambda X: float(string_list_initial_state[k]), boundary, component=k))
            print(f"float(string_list_initial_state[{k}]) = {float(string_list_initial_state[k])}")
            # Get the train data
            observe_y.append(dde.icbc.PointSetBC(observe_t, data_set_array[:, k+1:k+2], component=k))
            print(data_set_array[:, k+1:k+2])
            print("step 1")

        data_list = ic + observe_y
        data = dde.data.PDE(
            geom,
            ODE_model,
            data_list,
            num_domain=400,
            num_boundary=len(string_list_equations),
            anchors=observe_t,
            auxiliary_var_function=ex_func,
        )

        net = dde.nn.FNN([1] + [int(width_hidden_layers)] * int(number_hidden_layers) + [len(string_list_equations)], "tanh", "Glorot uniform")
        model = dde.Model(data, net)

        variable = dde.callbacks.VariableValue(
                external_trainable_variables, period=1000, filename="variables.dat"
            )

        # train adam
        model.compile(
            "adam", lr=0.001, external_trainable_variables=external_trainable_variables #, loss_weights=[1,1,1 , 1,1,1, 1,1,1]
        )
        losshistory, train_state = model.train(iterations=int(number_iterations), callbacks=[variable])

        # train lbfgs
        model.compile("L-BFGS", external_trainable_variables=external_trainable_variables) #, loss_weights=[1,1,1 , 1,1,1, 100,100,100])
        losshistory, train_state = model.train(callbacks=[variable])

        idx = np.argsort(train_state.X_test[:, 0])

        t = train_state.X_test[idx, 0]
        best_y = train_state.best_y

        LIST = best_y[idx, :]

        dft = pd.DataFrame(t, columns=['time'])
        dfy = pd.DataFrame(LIST)

        new_column_names_result = [f'result_col_{i+1}' for i in range(len(dfy.columns))]
        dfy.columns = new_column_names_result

        new_column_names_data = [f'data_col_{i+1}' for i in range(len(df_data.columns))]
        df_data.columns = new_column_names_data

        loss_steps = losshistory.steps
        loss_train = np.array([np.sum(loss) for loss in losshistory.loss_train])
        loss_test = np.array([np.sum(loss) for loss in losshistory.loss_test])

        df_loss_steps = pd.DataFrame(loss_steps, columns=['loss_steps'])
        df_loss_train = pd.DataFrame(loss_train, columns=['loss_train'])
        df_loss_test = pd.DataFrame(loss_test, columns=['loss_test'])

        df_time = pd.DataFrame(time, columns=['time_controller'])
        df_control = pd.DataFrame(ex_input, columns=['control'])

        print("step 5")
        df_result = pd.concat([dft, dfy, df_data, df_loss_steps, df_loss_test, df_loss_train, df_time, df_control], axis=1)

        print("step 6")

        df_result.to_csv("pinn_result.csv", index=False)

        print("step 7")

        ########################################################

        filename = 'variables.dat'

        # Read the last line and print
        cleaned_last_line = read_last_line_and_clean(filename)

        # Convert the cleaned string to a list
        K = convert_string_to_list(cleaned_last_line)

        idx = np.argsort(train_state.X_test[:, 0])

        best_y = train_state.best_y

        X_list = best_y[idx, 0]
        S_list = best_y[idx, 1]
        P_list = best_y[idx, 2]

        t = 0

        X, S, P = X_list[0], S_list[0], P_list[0]

        D_list = []

        while t < maxtime:

            print('##===============================================##')
            print(f'start step controller: time {t}')
            # S_setpoint = process.Scrit

            D = step_controller(S_setpoint=0.1,
                                X_meas=X,
                                S_meas=S,
                                P_meas=P,
                                t=t,
                                S_in=350,
                                horizon=control_horizon,
                                K=K)

            print(f"end step controller: time = {t}")
            print('__________________________________________________')
            D_list.append(D)
            print(f'D = {D}')
            print('__________________________________________________')

            print(f'start step model loop: time {t}')

            t2 = t + control_horizon

            while t < t2:
                measurements = step_model(D=D, X_=X, S_=S, P_=P,dt=dt, t_=t, K=K)
                print(f"step model: time = {t}")
                t = t + dt
                X, S, P = measurements

            print(f"end step model loop: time = {t}")
            print('__________________________________________________')

        print("end : all loops")

        print(f'Control law : {D_list}')

        u = uniform_step_function_generator(values_list=D_list, time=time)

        ################################################################################################
