from unicodedata import name
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import math


class GranularModules:
    def __init__(
            self, 
            refPlaneID: int,    # reference plane geom ID
            gndPlaneID: int,    # ground plane geom ID
            footIDs:    dict,   # dictionary of foot geom Names & IDs
            footNames:  list    # list of foot geom Names
        ) -> None:
        self.refPlaneID = refPlaneID
        self.gndPlaneID = gndPlaneID
        self.footIDs = footIDs
        self.footNames = footNames

        self.prev_zdot = {name: 0.0 for name in footIDs.keys()}
        self.prev_time = None
    

    def plotDataPlane2Foot(
            self,
            data: dict
        ) -> None:
        """
        Input:
            - data: dict containing recorded foot data of distances, velocities, accelerations
        Output:
            - saved plot in folder
        """
        
        if not data['time']:
            print("No data recorded to plot!")
            return

        tArray = np.array(data['time'])

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Foot Data w/ Respect to Reference Plane', fontsize=16)

        colors = {'FL':'red', 'FR':'blue', 'RL':'green', 'RR':'orange'}

        # Plot distances (z)
        axes[0].set_title('z of Foot (Normal Distances)')
        axes[0].set_ylabel('z(m)')
        for foot in self.footNames:
            if foot in data['z']:
                z = np.array(data['z'][foot])
                axes[0].plot(tArray, z, label=foot, color=colors[foot], linewidth=2)
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot velocities (z_dot)
        axes[1].set_title('z_dot of Foot (Normal Velocities)')
        axes[1].set_ylabel('z_dot(m/s)')
        for foot in self.footNames:
            if foot in data['z_dot']:
                z_dot = np.array(data['z_dot'][foot])
                axes[1].plot(tArray, z_dot, label=foot, color=colors[foot], linewidth=2)
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot accelerations (z_ddot)
        axes[2].set_title('z_ddot of Foot (Normal Accelerations)')
        axes[2].set_ylabel('z_ddot(m/s²)')
        axes[2].set_xlabel('Time (s)')
        for foot in self.footNames:
            if foot in data['z_ddot']:
                z_ddot = np.array(data['z_ddot'][foot])
                axes[2].plot(tArray, z_ddot, label=foot, color=colors[foot], linewidth=2)
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('foot_data_wr2_ref_plane.png', dpi=300, bbox_inches='tight')
        plt.show()
        

    def plot_GM_OnFoot(
            self,
            data: dict
        ) -> None:
        """
        Input:
            - data: dict containing recorded foot data of contact forces
        Output:
            - saved plot in folder showing z-component contact forces over time (one subplot per foot)
        """
        
        if not data['time']:
            print("No data recorded to plot")
            return

        tArray = np.array(data['time'])

        fig, axes = plt.subplots(4, 1, figsize=(12, 14))
        fig.suptitle('F_GM on Each Foot', fontsize=16)

        colors = {
            'FL':   'red', 
            'FR':   'blue', 
            'RL':   'green', 
            'RR':   'orange'
        }

        footFullName = {
            'FL': 'Front Left', 
            'FR': 'Front Right', 
            'RL': 'Rear Left', 
            'RR': 'Rear Right'
        }

        # Create a subplot for each foot's Fz component
        for idx, foot in enumerate(self.footNames):
            if foot in data['contact_force_vec']:
                forceVec = np.array(data['contact_force_vec'][foot])
                fz = forceVec[:, 2]
                
                axes[idx].plot(tArray, fz, color=colors[foot], linewidth=2, label=f'{foot} ({footFullName[foot]})')
                axes[idx].set_title(f'{foot} - {footFullName[foot]}', fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('Fz (N)', fontsize=11)
                axes[idx].legend(loc='upper right')
                axes[idx].grid(True, alpha=0.3)
                
                # Add zero reference line
                axes[idx].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        axes[-1].set_xlabel('Time (s)', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('foot_contact_forces.png', dpi=300, bbox_inches='tight')
        plt.show()
    

    def planeCenterNormal(
            self,
            mjData: mujoco.MjData
        ) -> tuple:
        """
        Input:
            - mjData: mujoco.MjData object
        Output:
            - normal vector & center of the plane
        """

        # Rotation Matrix of the reference plane
        R = np.array(mjData.geom_xmat[self.refPlaneID]).reshape(3, 3)

        # 3rd column of R = z-axis vector = normal vector
        n = R[:, 2]
        
        # Center of the plane
        p0 = mjData.geom_xpos[self.refPlaneID].copy()

        return n, p0
    

    def isFootFloorContact(
            self,
            mj_data: mujoco.MjData, 
            footID: int
        ) -> tuple:
        """
        Input:
            - mj_data:      mujoco.MjData object
            - foot_geom_id: geom ID of the foot
        Output:
            - whether the foot is in contact with the floor
            - minimum distance between the foot and the floor
        Note:
            - min_dist = infinity if no contact
        """

        minDist = None
        hasContact = False
        for k in range(mj_data.ncon):
            con = mj_data.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 == footID and g2 == self.gndPlaneID) or (g1 == self.gndPlaneID and g2 == footID):
                hasContact = True
                d = float(con.dist)
                minDist = d if minDist is None else min(minDist, d)

        if minDist is None:
            minDist = float("inf")
        
        return hasContact, minDist
    

    def distPlane2Foot(
            self,
            mjData:     mujoco.MjData, 
            monitor:    bool = False
        ) -> dict:
        """
        Input:
            - mjData:   mujoco.MjData object
            - monitor:  print the distances for monitoring purposes
        Output:
            - normal distances from the feet of Go2 to the plane
        Note:
            - Output = z
        """

        n, p0 = self.planeCenterNormal(mjData)

        distances = {}
        # Compute normal distance to reference plane for each foot
        for foot_name, foot_id in self.footIDs.items():
            # Get foot position
            foot_pos = mjData.geom_xpos[foot_id].copy()

            distance = (foot_pos - p0) @ n
            distances[foot_name] = distance

        if monitor:
            print("==================================")
            print(f"Foot z:      FL={distances['FL']:.4f}, FR={distances['FR']:.4f}, RL={distances['RL']:.4f}, RR={distances['RR']:.4f}")
        
        return distances

    def velAccPlane2Foot(
            self,
            mjModel:    mujoco.MjModel, 
            mjData:     mujoco.MjData,
            monitor:    bool = False
        ) -> dict:
        """
        Input:
            - mjModel:  mujoco.MjModel object
            - mjData:   mujoco.MjData object
            - monitor:  print the velocities & accelerations for monitoring purposes
        Output:
            - normal velocities & accelerations from the feet in global frame
        Note:
            - Output = z_dot, z_ddot
        """

        n, _ = self.planeCenterNormal(mjData)

        # compute dt
        dt = None
        t = float(mjData.time)
        if self.prev_time is not None:
            dt = t - self.prev_time
        else:
            dt = None

        velAcc = {}
        for foot_name, foot_id in self.footIDs.items():
            # end-effector(foot) jacobian
            jacp = np.zeros((3, mjModel.nv))
            _ = np.zeros((3, mjModel.nv)) # we do not need angular jacobian
            mujoco.mj_jacGeom(mjModel, mjData, jacp, _, foot_id)

            # linear velocity under global frame
            lin_v = jacp @ mjData.qvel
            # project to normal linear velocity
            z_dot = float(lin_v @ n)

            if dt is None or dt <= 0.0:
                z_ddot = 0.0
            else:
                z_ddot = (z_dot - self.prev_zdot[foot_name]) / dt

            velAcc[foot_name] = {"z_dot": z_dot, "z_ddot": z_ddot}
            self.prev_zdot[foot_name] = z_dot

        self.prev_time = t

        if monitor:
            print("----------------------------------")
            print("Foot z_dot:  " + ", ".join([f"{k}={v['z_dot']:.4f}" for k, v in velAcc.items()]))
            print("Foot z_ddot: " + ", ".join([f"{k}={v['z_ddot']:.4f}" for k, v in velAcc.items()]))

        return velAcc
    

    def modelNumericReader(
            self, 
            mjModel:    mujoco.MjModel, 
            name:       str
        ) -> float:
        """
        Input:
            - mjModel: mujoco.MjModel object
            - name:    name of the numeric parameter in the model
        Output:
            - value of the numeric parameter
        Note:
            - The numeric parameter must be scalar
            - Parameters are under <custom> tag: <numeric name=... data=...>
        """

        idx = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_NUMERIC, name)

        if idx < 0:
            raise KeyError(f"Missing <numeric name='{name}'> in model.")
        
        adr = mjModel.numeric_adr[idx]
        sz  = mjModel.numeric_size[idx]

        if sz != 1:
            raise ValueError(f"<numeric name='{name}'> must be scalar, got size={sz}.")
        
        return float(mjModel.numeric_data[adr])


    def get_GM_ParamsFromModel(
            self, 
            mjModel: mujoco.MjModel, 
            footID: int
        ) -> dict:
        """
        Input:
            - mjModel: mujoco.MjModel object
            - footID:  geom ID of the foot
        Output:
            - dictionary of F_GM parameters for the foot
        """
        
        # Intruder Radius
        r = float(mjModel.geom_size[footID, 0])

        if r <= 0:
            raise ValueError(f"Invalid foot radius: {r}")
        
        A = math.pi * r * r     # Intruder cross-sectional area
        P = 2.0 * math.pi * r   # Intruder perimeter

        # Other parameters from env model
        return dict(
            A=      A,
            P=      P,
            theta=  self.modelNumericReader(mjModel, "gm_theta"),
            nu=     self.modelNumericReader(mjModel, "gm_nu"),
            z0=     self.modelNumericReader(mjModel, "gm_z0"),
            phi=    self.modelNumericReader(mjModel, "gm_phi"),
            rho=    self.modelNumericReader(mjModel, "gm_rho"),
            c_g=    self.modelNumericReader(mjModel, "gm_cg"),
            c_d=    self.modelNumericReader(mjModel, "gm_cd"),
            sigma_flat= self.modelNumericReader(mjModel, "gm_sigma_flat"),
            epsilon_f=  self.modelNumericReader(mjModel, "gm_eps_f"),
            sigma_cone=(
                self.modelNumericReader(mjModel, "gm_sigma_cone")
                if mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_NUMERIC, "gm_sigma_cone") >= 0
                else 0.0
            ),
            ema_alpha_base= 0.8, # EMA smoothing factor
            ema_rate_c_r=   0.0,
        )


    def compute_GM_SingleFoot(
            self,
            z:          float,
            z_dot:      float,
            z_ddot:     float,
            params:     dict
        ) -> float:
        """
        Input:
            - z:            penetration depth (>=0)
            - z_dot:        penetration velocity (>=0)
            - z_ddot:       penetration acceleration
            - params:       dictionary of F_GM parameters for the foot
        Output:
            - F_GM force of one single foot
        """

        # parameters
        A =     float(params["A"])
        P =     float(params["P"])
        theta = float(params["theta"])
        nu =    float(params["nu"])
        z0 =    float(params.get("z0", 0.0))
        phi =   float(params["phi"])
        rho =   float(params["rho"])
        c_g =   float(params["c_g"])
        c_d =   float(params["c_d"])
        sigma_flat =        float(params["sigma_flat"])
        sigma_cone =        float(params.get("sigma_cone", 0.0))

        dz = z - z0
        rh = 2.0 * A / P
        alpha = nu / math.tan(theta)

        # surface area of the developing cone & its integral
        A_cone = 0.0
        I_cone = 0.0 # I_cone = integral of A_cone
        
        core = rh - alpha * dz
        if core > 0.0:
            A_flat = math.pi * core * core
            I_flat = math.pi * ((rh*rh)*dz - rh*alpha*(dz**2) + (alpha*alpha)*(dz**3)/3.0)
        else:
            A_flat = 0.0
            I_flat = 0.0 # I_flat = integral of A_flat

        # added-mass
        m_a     = c_g * phi * rho * nu * I_flat
        m_a_dot = (c_g * phi * rho * nu * A_flat) * z_dot # (dm_a/dz)* (dz/dt)

        # quasistatic force
        F_p   = sigma_flat * I_flat + sigma_cone * I_cone

        # raw F_GM
        F_out_raw = F_p - c_d * m_a_dot * z_dot - m_a * z_ddot

        # z > 0 means foot is penetrating
        F_out = max(0.0, F_out_raw if z > 0.0 else 0.0)

        return F_out


    def compute_GM_AllFoot(
            self,
            mjModel:        mujoco.MjModel,
            mjData:         mujoco.MjData,
            paramsPerFoot:  dict | None = None,
            monitor:        bool = False,
        ) -> dict:
        """
        Input:
            - mjModel:      mujoco.MjModel object
            - mjData:       mujoco.MjData object
            - paramPerFoot: dict of customized params for each foot, or None to use FL's params for all
            - monitor:      print the computed forces for monitoring purposes
        Output:
            - dictionary of foot names with respoective F_GM {foot_name: np.array([Fx,Fy,Fz])} in global frame
        Note:
            - paramPerFoot dict example: {'FL': params_FL, ...}
        """

        eps_depth = 1e-5
        eps_speed = 1e-5
        delta = 0.001  # distance for force fade-out

        # normal vector of reference plane
        n, _ = self.planeCenterNormal(mjData)
        n = np.asarray(n, float)

        # extract z, z_dot, z_ddot
        z_dict =    self.distPlane2Foot(mjData, monitor=False)
        va_dict =   self.velAccPlane2Foot(mjModel, mjData, monitor=False)

        # configure custom foot params if not given
        if paramsPerFoot is None:
            if "FL" not in self.footIDs:
                any_name = next(iter(self.footIDs.keys()))
                base_gid = self.footIDs[any_name]
            else:
                base_gid = self.footIDs["FL"]
            base_params = self.get_GM_ParamsFromModel(mjModel, base_gid)
            paramsPerFoot = {name: base_params for name in self.footIDs.keys()}

        # compute F_GM for each foot
        forces = {}
        for name, _ in self.footIDs.items():
            z_n =         float(z_dict[name])                 
            z_dot_n =     float(va_dict[name]["z_dot"])
            z_ddot_n =    float(va_dict[name]["z_ddot"])

            if (z_n < -eps_depth) and (z_dot_n < -eps_speed):
                # invert sign: downwards = positive
                z =         max(0.0, -z_n)          # penetration depth
                z_dot =     max(0.0, -z_dot_n)      # penetration velocity
                z_ddot =    -z_ddot_n               # penetration acceleration

                Fn_raw = self.compute_GM_SingleFoot(
                    z, 
                    z_dot, 
                    z_ddot,
                    params=paramsPerFoot[name]
                )

                # fade-out when close to the surface
                hasContact, dist = self.isFootFloorContact(mjData, self.footIDs[name])
                if hasContact:
                    s = dist / max(1e-9, delta)
                    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
                    w_floor = s*s*(3.0 - 2.0*s)
                else:
                    w_floor = 1.0

                Fn = max(0.0, Fn_raw) * w_floor

                forces[name] = Fn * n
            else:
                # If foot is not penetrating, set force to zero
                forces[name] = np.zeros(3)

        if monitor:
            line = ", ".join([f"{k}={np.dot(v,n):.2f}" for k,v in forces.items()])
            print(f"[GM] Fn (along n): {line}")

        return forces