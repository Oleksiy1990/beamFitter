    def fitandplot_2D(self,rotation=False):
        
        if self.fit2Dparams == None:
            self.__fit2D(rotation=rotation)
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        X,Y = self.x2Dgrid,self.y2Dgrid
        if rotation:
            fitted_surface_beam = residual_G2D(self.fit2Dparams,X,Y)
        else:
            fitted_surface_beam = residual_G2D_norotation(self.fit2Dparams,X,Y)
        original_surface_beam = self.formatted_array
        
        ax.plot_surface(X, Y, fitted_surface_beam, cmap=cm.bwr,\
                       linewidth=0, antialiased=False)
        ax.plot_surface(X,Y,original_surface_beam,cmap=cm.bwr,linewidth=0,antialiased=False)

        #cset = ax.contour(X, Y, fitted_surface_beam, zdir='z',offset=0, cmap=cm.bwr)
        cset = ax.contour(X, Y, fitted_surface_beam, zdir='x',\
            offset=0, cmap=cm.bwr)
        cset = ax.contour(X, Y, fitted_surface_beam, zdir='y',\
            offset=0, cmap=cm.bwr)

        #cset1 = ax.contour(X, Y, original_surface_beam, zdir='z',offset=0, cmap=cm.bwr)
        cset1 = ax.contour(X, Y, original_surface_beam, zdir='x',\
            offset=0, cmap=cm.bwr)
        cset1 = ax.contour(X, Y, original_surface_beam, zdir='y',\
            offset=0, cmap=cm.bwr)
        #fig.colorbar(surf)

        plt.show()



            def __fit2D(self,minim_method="leastsq",rotation=False,initial_params_2D = None):
        if initial_params_2D is None:
            self.__fit_axis(0,minim_method)
            self.__fit_axis(1,minim_method)

            # we first take all the initial parameters from 1D fits
            bgr2D_est = np.mean([self.axis0fitparams.valuesdict()["backgr"]/len(self.axis0pts),self.axis1fitparams.valuesdict()["backgr"]/len(self.axis1pts)])
            x2D_est = self.axis0fitparams.valuesdict()["r_zero"]
            omegaX2D_est = self.axis0fitparams.valuesdict()["omega_zero"]
            y2D_est = self.axis1fitparams.valuesdict()["r_zero"]
            omegaY2D_est = self.axis1fitparams.valuesdict()["omega_zero"]

            smoothened_image = gaussian_filter(self.image_array,50)
            peakheight2D_est = np.amax(smoothened_image)
            #now we need to programatically cut out the region of interest out of the
            #whole picture so that fitting takes way less time

            # NOTE! In this implementation, if the beam is small compared to picture size
            # and is very close to the edge, the fitting will fail, because the x and y
            # center position estimates will be off

            self.__format_picture(x2D_est,omegaX2D_est,y2D_est,omegaY2D_est)
            cropped_data = self.formatted_array
            xvals = np.linspace(1,cropped_data.shape[0],cropped_data.shape[0])
            yvals = np.linspace(1,cropped_data.shape[1],cropped_data.shape[1])
            x, y = np.meshgrid(yvals,xvals)
            # NOTE! there's apparently some weird convention, this has to do with
            # Cartesian vs. matrix indexing, which is explain in numpy.meshgrid manual

            estimates_2D = Parameters()
            estimates_2D.add("I_zero",value=peakheight2D_est,min=bgr2D_est)
            estimates_2D.add("x_zero",value=0.5*len(yvals),min=0,max=len(yvals)) # NOTE! weird indexing conventions
            estimates_2D.add("y_zero",value=0.5*len(xvals),min=0,max=len(xvals)) # NOTE! weird indexing conventions
            estimates_2D.add("omegaX_zero",value=omegaX2D_est)
            estimates_2D.add("omegaY_zero",value=omegaY2D_est)
            estimates_2D.add("theta_rot",value=0.578,min = 0,max = np.pi/2) #just starting with 0
            estimates_2D.add("backgr",value=bgr2D_est)
            print("Here are the parameters before the start of the fit: ")
            print(estimates_2D.valuesdict())
        else:
            estimates_2D = initial_params_2D
            print("Here are the parameters before the start of the fit: ")
            print(estimates_2D.valuesdict())

        if rotation:
                fit2D = Minimizer(residual_G2D,estimates_2D,fcn_args=(x,y),fcn_kws={"data":cropped_data})
                print("Including rotation")
        else:
            fit2D = Minimizer(residual_G2D_norotation,estimates_2D,fcn_args=(x,y),fcn_kws={"data":cropped_data})
            print("Not including rotation")

        fit_res2D = fit2D.minimize(minim_method)

        self.x2Dgrid = x
        self.y2Dgrid = y
        self.fit2Dparams = fit_res2D.params
        # return (x,y,fit_res2D)

            
    def fitandprint_2D(self,rotation=False,save_json = False):
        if self.fit2Dparams == None:
            self.__fit2D(rotation=rotation)
        print("The sizes are in px")
        if save_json:
            json_file = self.pathbase+".json"
            with open(json_file,"w") as outfile:
                json.dump(self.fit2Dparams.valuesdict(),outfile,indent=4)
        for (key,val) in self.fit2Dparams.valuesdict().items():
            print(key,"=",val)
