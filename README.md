# Glacier Deep Inversion
A project to enable the inversion of geophysical parameters of glaciers using Deep Learning methods.

---
the experience file is splitted in several sections :

{
    "batch_size": integer
    "epochs": integer
    "min_learning_rate": float (minimum at which the learning rate should decrease)
    "patience": integer (how much epoch of no improvement before decreasing learning rate)
    "optimizer_name": string
    "criterion": string (name of value on which "improvement" is based)
    "model_name": string
    "model_architecture": string (which model described in ,
    "area_size": 400.0,
    "seeds_number": 30,

    "inputs": list of dictionaries (example : {
        "name": string (name of the variable)
        "resolution": float (pixel size resolution)
        "scalar": boolean (if true, will extract the center value of the image. if false, keep it as an image)
        
    }

    "outputs": list of dictionaries (same as "inputs")

    "losses_name": dictionary
    "metrics_name": dictionary
}

"drawing_description": dictionary (example : {
	"features": list (which entities in the shapefile dataset to perform estimation)
	"profiles": list
	"models": list (what is the name of the model)
})

"dataset_description": {
	"dataset_name": string (will be used as the name of the folder)
	"epsg": integer (coordinate system id, ex: 2056)
	"shapefile_file": string (which shapefile in data to use, add ".shp" at the end),
	"shapefile_fieldname": string (which column is used as id of entities in the shapefile),
	"buffer_distance": float (how much extra space to add when extracting glacier images)
	"rasters": list of dictionaries (which raster images to be used, example : {
		"name" : string (the name of the variable)
		"resolution": float (pixel size at which it will resample)
		"file": string (name of the raster file, add ".tif" at the end)
	})
}

"model_description": {
	"inputs": dictionary (name of the variable as key and dictionary of parameters as value)
	"main": dictionary (same)
	"outputs": dictionary (same)
}
