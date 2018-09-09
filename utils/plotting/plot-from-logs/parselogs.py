import sys
import re
import matplotlib.pyplot as plt

def parse_log_files(*log_file_path_list):
    overall_loss_dict = {}
    bbox_loss_dict = {}
    cls_loss_dict = {}
    rpn_cls_loss_dict = {}
    rpn_bbox_loss_dict = {}

    iters = 0
    log_file_path_list = log_file_path_list[0]


    for i in range(len(log_file_path_list)):
        file_path = log_file_path_list[i]
        
        log_file = open(file_path, 'r')
        log = log_file.read()
        log_file.close()

        log_lines = log.split('\n')
        loss_lines = filter(lambda x : "Train net output" in x or "Iteration" in x, log_lines)
        loss_lines = loss_lines[1:]

        for j in range(0, len(loss_lines), 6):
            m = re.search('.*(Iteration )([\d]*).*(loss = )([\d\.]*)$', loss_lines[j+0])
            
            iteration_num = int(m.group(2)) + iters
            
            #print("Handling iteration number {0}".format(iteration_num))
            loss = m.group(4)
            overall_loss_dict[iteration_num] = float(loss)

            m = re.search('.*(loss_bbox = )([\d\.]*).*$', loss_lines[j+1])
            bbox_loss = m.group(2)
            bbox_loss_dict[iteration_num] = float(bbox_loss)

            m = re.search('.*(loss_cls = )([\d\.]*).*$', loss_lines[j+2])
            cls_loss = m.group(2)
            cls_loss_dict[iteration_num] = float(cls_loss)

            m = re.search('.*(rpn_cls_loss = )([\d\.]*).*$', loss_lines[j+3])
            rpn_cls_loss = m.group(2)
            rpn_cls_loss_dict[iteration_num] = float(rpn_cls_loss)

            m = re.search('.*(rpn_loss_bbox = )([\d\.]*).*$', loss_lines[j+4])
            rpn_loss_bbox = m.group(2)
            rpn_bbox_loss_dict[iteration_num] = float(rpn_loss_bbox)

        iters = iteration_num

    plot_and_save_fig(overall_loss_dict, "Iteration num", "Total loss", "Total loss vs Iterations", "total_loss.png")
    plot_and_save_fig(bbox_loss_dict, "Iteration num", "BBox loss", "BBox loss vs Iterations", "bbox_loss.png")
    plot_and_save_fig(bbox_loss_dict, "Iteration num", "CLS loss", "CLS loss vs Iterations", "cls_loss.png")
    plot_and_save_fig(rpn_cls_loss_dict, "Iteration num", "RPN CLS loss", "RPN CLS loss vs Iterations", "rpn_cls_loss.png")
    plot_and_save_fig(rpn_cls_loss_dict, "Iteration num", "RPN BBOX loss", "RPN BBOX loss vs Iterations", "rpn_bbox_loss.png")


def plot_and_save_fig(coordinates_dict, x_label, y_label, title, filename):
    plt.figure(figsize=(25,7))

    coords = sorted(coordinates_dict.iteritems())

    X = map(lambda (x,y) : x, coords)
    Y = map(lambda (x,y) : y, coords)

    plt.plot(X, Y, 'o--')
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    xticks = range(X[0], X[-1], (X[-1] - X[0])/30)

    plt.xticks(xticks, rotation='vertical')
    
    plt.savefig(filename)
    plt.close()



if __name__ == "__main__":
    log_file_paths = sys.argv[1:]
    
    parse_log_files(log_file_paths)



    