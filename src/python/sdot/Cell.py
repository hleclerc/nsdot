from .aggregate import aggregate, Tensor, ShapeVar, Axis


@aggregate
class Cell:
    nb_dims      = ShapeVar() # nb_dims est un tenseur entier de rang 0
    nvec         = Axis( 2 * nb_dims + 1 ) # le 1er argument donne la taille max selon cet axe. À ce stade c'est une opération symbolique
    dim          = Axis( nb_dims )
    frame        = Tensor( nvec, dim ) # on doit pouvoir trouver `nb_dims` tel que `frame.shape = [ nb_dims + 1, nb_dims ]`
