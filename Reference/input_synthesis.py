import numpy as np
import PIL.Image as Image
import matplotlib as mpl
import matplotlib.pylab as plt
from keras import backend as K
import time






def synthesize(model, x_original, suspicious_indices, step_size, d):

    input_tensor = model.input

    perturbed_set_x = []
    perturbed_set_y = []
    original_set_x  = []
    #print(len(x_original))
    q = 0
    for x in x_original:
        all_grads = []
        start_time = time.time()

        for s_ind in suspicious_indices:

            loss = K.mean(model.layers[s_ind[0]].output[..., s_ind[1]])
            grads = K.gradients(loss, input_tensor)[0]
            iterate = K.function([input_tensor], [loss, grads])
            _, grad_vals = iterate([np.expand_dims(x, axis=0)])
            all_grads.append(grad_vals[0])
        elapsed_time = time.time() - start_time
        #print("synthesis gradient part time is ", elapsed_time)
        perturbed_x = x.copy()



        start_time = time.time()
        for i in range(x.shape[1]):

            for k in range(x.shape[2]):

                sum_grad = 0
                for j in range(len(all_grads)):
                    sum_grad += all_grads[j][0][i][k]
                avg_grad = float(sum_grad) / len(suspicious_indices)
                avg_grad = avg_grad * step_size

                # Clipping gradients.
                if avg_grad > d:
                    avg_grad = d
                elif avg_grad < -d:
                    avg_grad = -d

                perturbed_x[0][i][k] = max(min(x[0][i][k] + avg_grad, 1), 0)
                # perturbed_x.append(max(min(x[0][i][k] + avg_grad, 1), 0))
        elapsed_time = time.time() - start_time
        #print("synthesis image pixel time is ", elapsed_time)
        '''
        for i in range(len(flatX)):
            sum_grad = 0
            for j in range(len(all_grads)):
                sum_grad += all_grads[j][0][i]
            avg_grad = float(sum_grad) / len(suspicious_indices)
            avg_grad = avg_grad * step_size
            if avg_grad > d:
                avg_grad = d
            elif avg_grad < -d:
                avg_grad = -d
            perturbed_x.append(max(min(flatX[i] + avg_grad, 1), 0))
        '''

        perturbed_set_x.append(perturbed_x)
        # perturbed_set_y.append(y)
        # original_set_x.append(x)
        
        #img = Image.fromarray(perturbed_set_x[0][0], 'F')
        #img.show()
        #plt.savefig(np.array(perturbed_set_x[0]))
    return perturbed_set_x
