# get args
force_redo=false
batch_size=1

while [ $# -gt 0 ]; do
    case "$1" in
        --number_of_images)
            number_of_images="${2#*=}"
            shift 2
            ;;
        --file_to_read)
            file_to_read="${2#*=}"
            shift 2
            ;;
        --image_mode)
            image_mode="${2#*=}"
            shift 2
            ;;
        --dataset_mode)
            dataset_mode="${2#*=}"
            shift 2
            ;;
        --dataset_mode)
            dataset_mode="${2#*=}"
            shift 2
            ;;
        --image_dir)
            image_dir="${2#*=}"
            shift 2
            ;;
        --narr_dir)
            narr_dir="${2#*=}"
            shift 2
            ;;
        --force_redo)
            force_redo=true
            shift 1
            ;;
        --batch_size)
            batch_size="${2#*=}"
            shift 2
            ;;
        *)
            printf "Invalid arg\n"
            exit 1
    esac
done

if [[ $force_redo = true ]]; then
    rm -r data
    rm data_stuff/downloader.py
fi

if ! test -f "data/dataset.jsonl"; then
    mkdir data
    # download image downloader
    wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py -P data_stuff

    # download narrative dataset
    wget https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl -O $narr_dir

    # make image ids
    python data_stuff/make_ids.py 0 $number_of_images $narr_dir $dataset_mode

    # download image files
    python data_stuff/downloader.py data/file_ids.txt --download_folder $image_dir

    # make a dataset
    python data_stuff/combine_datasets.py $image_dir $narr_dir data/dataset.jsonl
fi

# train
python train.py $batch_size