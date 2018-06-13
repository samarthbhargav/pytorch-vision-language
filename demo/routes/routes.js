module.exports = function (express) {
    var router = express.Router();
    var PythonShell = require('python-shell');
    var api_root = '/api';
    var path = require('path');

    router.get(api_root, function (req, res) {
        res.json({ 'success' : true, 'message' : 'Welcome', 'status' : 'Online' });
    });

    // host images
    router.get('/images/*', function (req, res) {
        var path_list = req.url.split('/');
        // path_list.shift(); path_list.shift();
        res.sendFile(path.join(__dirname, '../../data/', path_list.join('/')));
    });

    return router;
};
