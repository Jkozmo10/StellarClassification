
class Body:

    def __init__(self, objId, alpha, delta, u, g, r, i, z, run, rerun, cam, field, specObj, classification, redshift, plate, MJD, fiber):
        self.objId = objId
        self.alpha = alpha
        self.delta = delta
        self.u = u
        self.g = g
        self.r = r
        self.i = i
        self.z = z
        self.run = run
        self.rerun = rerun
        self.cam = cam
        self.field = field
        self.specObj = specObj
        self.classification = classification
        self.redshift = redshift
        self.plate = plate
        self.MJD = MJD
        self.fiber = fiber

    def __str__(self):
        return "objId " + self.objId + " alpha " + self.alpha


