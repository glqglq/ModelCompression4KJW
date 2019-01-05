for file in `ls *.tar`
do 
    todir=`echo $file | cut -d"." -f1`
    sudo mkdir $todir
    sudo tar -xvf $file -C $todir
done 