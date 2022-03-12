<?php
	if ($_POST['code'] == 'update_data'){
        file_put_contents('./trained_data/w.json', $_POST['wdata']);
        file_put_contents('./trained_data/b.json', $_POST['bdata']);
        
        echo "👍";
    }
?>