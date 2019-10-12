import visdom
import numpy as np
import ipdb

class Dashboard(object):

    def __init__(self, server=None, port=None, env='main'):

        super(Dashboard, self).__init__( )
        self.server = server
        self.port = port
        self.env = env
        self.vis = visdom.Visdom(server=server, port=self.port, env=self.env)

    def show_curve(self, train_data, val_data, datatype):

        iteration = range(len(train_data))

        x_axis = np.stack((np.array(iteration),
                           np.array(iteration))).transpose()

        y_axis = np.stack((np.array(train_data),
                           np.array(val_data))).transpose()

        train_type = 'train_{}'.format(datatype)
        val_type = 'val_{}'.format(datatype)
        self.vis.line(Y=y_axis,
                      X=x_axis,
                      win=datatype,
                      env=self.env,
                      opts=dict(legend = [train_type, val_type],
                                showlegend = True,
                                xlabel = 'epoch',
                                ylabel = datatype,
                                title = datatype))

    def show_sig_curve(self, data, datatype):
        x_axis = np.array(list(data.keys()))
        y_axis = np.array(list(data.values()))

        self.vis.line(Y=y_axis,
                      X=x_axis,
                      win=datatype,
                      env=self.env,
                      opts=dict(legend = [datatype],
                                showlegend = True,
                                xlabel = 'epoch',
                                ylabel = datatype,
                                title = datatype))

    def show_img(self, img, datatype):

        self.vis.image(img=img, win=datatype, opts=dict(legend=datatype,
                                                        title=datatype))

    def show_feature_maps(self, features, datatype):
        #ipdb.set_trace()
        # features = features.squeeze(dim=0)
        #features = features[:, None, :, :]

        self.vis.images(tensor=features, nrow=2, win=datatype, opts=dict(legend=datatype, title=datatype))


if __name__ == '__main__':

    vis = Dashboard(server='http://127.0.0.1', port=8011, env='example')

    train = {}
    val = {}

    for i in range(1000):

        train[i] = np.sqrt(i)
        val[i] = np.log(i+1)

    vis.show_curve(train_data=train, val_data=val, datatype='example')
    vis.show_sig_curve(data=train, datatype='y')


    for i in range(1000):
        img = np.random.random((500, 600, 3))
        img = np.round(img*255)
        img = np.array(img)
        img = img.transpose(2, 0, 1)

        # vis.show_img(img, datatype='img')